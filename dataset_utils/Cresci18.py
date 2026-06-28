import os
from warnings import warn
from zipfile import ZipFile
import csv
from dataset_utils.constants import Sample, UserData, TweetData
from torch.utils.data import IterableDataset
from dataset_utils.splitting import hash_split_multi
import sqlite3  




class Cresci18(IterableDataset):   
    """
    Dataset for the Cresci18. The downloaded *.csv.zip files can be in the directory /datasets 
                              or in another dataset whose filepath needs to be given as param to the constructor.
    """
    def __init__(self, mode :str, train_split: float = 0.8, dev_split: float = 0.1, root : str |None = None, use_only_labelled = True, label_mapping = ["bot","human","unlabelled"], sub_sample_size = -1):
        """
        :param root(OPTIONAL): the filepath of the dataset directory in which the dataset is stored, if none is given "datasets"
        :param mode: Dataset split to use ("train", "dev", or "test").
        :param train_split: Fraction of users assigned to the training set (e.g. 0.8 = 80%).
        :param dev_split: Fraction of users assigned to the validation set (e.g. 0.1 = 10
        :param use_only_labelled: if only users that have been categorized into 'bot' or 'human' should be provided
        :param label_mapping: List of labels to use for provided samples. Format: `['bot', 'human', 'unlabelled']`
        :param sub_sample_size: (Default: -1) the upper limit of the amount samples the iterator should provide. Fewer samples are possible if the dataset doesn't have enough samples.
        """
         #__init__ does the filehandling
        if root is None:
            root = "datasets"

        self.sub_sample_size = sub_sample_size
        self.mode = mode
        self.use_only_labelled = use_only_labelled
        self.label_mapping = {
            '0': label_mapping[1],
            '1': label_mapping[0],
            '':  label_mapping[2],
        }

        assert train_split + dev_split < 1.0

        extract_dir = "Cresci18"
        
        if not os.path.exists(os.path.join(root, extract_dir)):
            user_zip_path = os.path.join(root, "users.csv.zip")
            tweets_zip_path = os.path.join(root, "tweets.csv.zip")
            if not os.path.exists(user_zip_path):
                warn(
                "Dataset Cresci18's user.csv.zip file not found. Please make sure 'user.csv.zip' is in the dataset directory.",                    UserWarning
                )
                return
            
            if not os.path.exists(tweets_zip_path):
                warn(
                "Dataset Cresci18's tweets.csv.zip' file not found. Please make sure 'tweets.csv.zip' is in the dataset directory.",                    UserWarning
                )
                return

            with ZipFile(tweets_zip_path) as zipObj:
                zipObj.extractall(os.path.join(root,extract_dir))
            os.remove(tweets_zip_path) # removes the zip file after extraction to save space
            print("Finished 'tweets.csv.zip' extraction and removed the zip file")

            with ZipFile(user_zip_path) as zipObj:
                zipObj.extractall(os.path.join(root, extract_dir))
            os.remove(user_zip_path) # removes the zip file after extraction to save space
            print("Finished 'users.csv.zip' extraction and removed the zip file")
        
        self.user_data_path = os.path.join(root, extract_dir, "users.csv" )
        self.tweets_data_path = os.path.join(root, extract_dir, "tweets.csv" )


        if not os.path.exists(self.user_data_path):
            warn(
                "Dataset Cresci18's users.csv file not found. Please make sure 'users.csv' is in the Cresci18 directory.",                    UserWarning
                )
            return
        if not os.path.exists(self.tweets_data_path):
            warn(
                "Dataset Cresci18's tweets.csv' file not found. Please make sure 'tweets.csv' is in the Cresci18 directory.",                    UserWarning
                )
            return


    def __iter__(self):
            """Reads the users metadata and the tweetdata. 
            
            Then joins on user_id using an on-disk SQLite buffer for sorting.
            """
            iteration = -self.sub_sample_size
            # 1. Load users into memory
            users = {}
            labels = {}
            with open(self.user_data_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=",")
                for row in reader:
                    label = row["bot"]
                    if self.use_only_labelled and not label in ['0','1']: continue

                    users[row["id"]] = UserData.from_row(row)
                    labels[row["id"]] = label
            # 2. Get CSV headers from tweets file to dynamically map SQLite columns
            with open(self.tweets_data_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=",")
                headers = reader.fieldnames

            if not headers:
                raise ValueError(f"The CSV file at {self.tweets_data_path} is empty or missing headers.")
            
            headers = list(headers)  # Ensure it's a mutable list for indexing

            # 3. Setup a clean, temporary SQLite database (NO row_factory used here)
            db_path = os.path.join(os.path.dirname(self.tweets_data_path), "temp_tweets_sort.db")
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                except Exception:
                    pass

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Optimize SQLite speed configurations
            cursor.execute("PRAGMA synchronous = OFF")
            cursor.execute("PRAGMA journal_mode = MEMORY")
            
            # Fresh slate table creation
            cursor.execute("DROP TABLE IF EXISTS tweets")
            columns_def = ", ".join([f'"{h}" TEXT' for h in headers])
            cursor.execute(f"CREATE TABLE tweets ({columns_def})")

            columns_str = ", ".join([f'"{h}"' for h in headers])
            placeholders = ", ".join(["?"] * len(headers))
            insert_query = f"INSERT INTO tweets ({columns_str}) VALUES ({placeholders})"

            # 4. Stream from CSV -> Apply Filters -> Populate SQLite in Chunks
            batch = []
            batch_size = 50000

            with open(self.tweets_data_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=",")
                for row in reader:
                    user_id = row.get("user_id")
                    if not user_id:
                        continue

                    # Filter early: Split check
                    if hash_split_multi(user_id) != self.mode:
                        continue

                    # Filter early: Active User check
                    if user_id not in users:
                        continue

                    # Append row data as an ordered tuple matching our headers schema
                    batch.append([row.get(h, "") for h in headers])

                    if len(batch) >= batch_size:
                        cursor.executemany(insert_query, batch)
                        conn.commit()
                        batch = []

                if batch:
                    cursor.executemany(insert_query, batch)
                    conn.commit()

            # Build an index to speed up the disk-sorting query step
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON tweets (user_id)")
            conn.commit()

            # 5. Stream SORTED data out of SQLite as standard tuples
            cursor.execute("SELECT * FROM tweets ORDER BY user_id")
            
            user_id_idx = headers.index("user_id")
            last_user = None
            current_tweets = []

            for row in cursor:
                # Safely grab user_id from the tuple index
                user_id = row[user_id_idx]

                if last_user is not None and user_id != last_user:
                    if current_tweets:
                        user_obj = users.get(last_user)
                        if not user_obj:
                            raise ValueError(
                            f"The CSV file at {self.tweets_data_path} is empty or missing headers."
                        )
                        #current_label = getattr(user_obj, "label", "")
                        current_label = self.label_mapping.get(labels[last_user], self.label_mapping[''])
                        yield Sample(
                            tweet_data=current_tweets,
                            user_data=user_obj,
                            label=current_label,
                        )
                        # count and stop early if only a sub sample is requested
                        if self.sub_sample_size > 0:
                            iteration += 1
                            if iteration >= 0: break
                    current_tweets = []

                last_user = user_id
                
                # FOOLPROOF DICTIONARY CONVERSION: Map headers list to tuple values
                row_dict = dict(zip(headers, row))
                
                # Pass a standard Python dictionary to your dataclass
                current_tweets.append(TweetData.from_row(row_dict))

            # Flush the final user block remaining in the buffer
            if current_tweets and last_user is not None:
                user_obj = users.get(last_user)#
                if not user_obj:
                            raise ValueError(
                            f"The CSV file at {self.tweets_data_path} is empty or missing headers."
                        )
                #current_label = getattr(user_obj, "label", "")
                current_label = self.label_mapping.get(labels[last_user], self.label_mapping[''])
                yield Sample(
                    tweet_data=current_tweets,
                    user_data=user_obj,
                    label=current_label,
                )

            # 6. Database Connection Cleanup
            conn.close()
            try:
                os.remove(db_path)
            except Exception:
                pass

        
       
 

       
        
    





        