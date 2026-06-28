import shutil
import csv
import json
import ijson
import sys
import os
from warnings import warn
import sqlite3

try:
    from dataset_utils.constants import UserData, TweetData, Sample, normalize_user_id, json_decimal_handler
except ImportError:
    from dataset_utils.constants import UserData, TweetData, Sample, normalize_user_id, json_decimal_handler

from torch.utils.data import IterableDataset
from dataset_utils.splitting import hash_split_multi

class Twibot22(IterableDataset):
    """
    Dataset for the Twibot-22. The downloaded *.json files or a directory,named "Twibot22", containing them, need to be in the directory /datasets.
    IMPORTANT: IF YOU CAN DOWNLOAD THEM DIRECTLY INTO THE TWIBOT22 FOLDER.
    """
    def __init__(self, mode :str, train_split: float = 0.8, dev_split: float = 0.1, root : str |None = None, label_mapping = ["bot","human"]):
        """
        :param root: the filepath of the dataset directory in which the dataset is stored.
        :param mode: Dataset split to use ("train", "dev", or "test").
        :param train_split: Fraction of users assigned to the training set (e.g. 0.8 = 80%).
        :param dev_split: Fraction of users assigned to the validation set (e.g. 0.1 = 10
        :param label_mapping: List of labels to use for provided samples. Format: `['bot', 'human', 'unlabelled']`
        """
        if ijson.backend_name == 'python':
            print("\n" + "="*70)
            print("⚠️  WARNING: ijson is running in ultra-slow pure-Python mode!")
            print("Parsing these multi-gigabyte files will take ages.")
            print("\nHow to fix this:")
            if sys.platform.startswith('win'):
                print(" -> Open Anaconda Prompt and run: conda install -c conda-forge yajl")
            elif sys.platform.startswith('linux'):
                print(" -> Run: sudo apt-get install libyajl2")
            elif sys.platform.startswith('darwin'): # Mac
                print(" -> Run: brew install yajl")
            print("="*70 + "\n")

        #__init_ does the file handling
        if root == None:
            root = "datasets"

        self.mode = mode
        self.label_mapping = {
            'bot': label_mapping[0],
            'human': label_mapping[1],
        }

        assert train_split + dev_split < 1.0

        # creates subdirectory "Twibot22" if it doesn't exist 
        if not os.path.exists(os.path.join(root, "Twibot22")):
             os.makedirs(os.path.join(root, "Twibot22"), exist_ok=True)

        #moves all json files into the "Twibot22" subdirectory
        for file in os.listdir(root):
            if file.endswith(".json") | file.endswith(".csv"):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(root, "Twibot22", file)
                shutil.move(src_path, dst_path)

        #defines the paths to the user data, labels and tweets
        self.user_data_path = os.path.join(root, "Twibot22", "user.json")
        self.labels_path = os.path.join(root, "Twibot22", "label.csv")
        self.tweet_paths = [
            os.path.join(root, "Twibot22", f"tweet_{i}.json")
            for i in range(9)
        ]


        #validation of the expected paths 
        if not os.path.exists(self.user_data_path):
            warn(
                "Dataset 'Twibo22.user.json' file not found. Please make sure 'user.json' is in the dataset directory.",                    UserWarning
            )
            return

        if not os.path.exists(self.labels_path):
            warn(
                "Dataset 'Twibot22.label.tsv' file not found. Please make sure 'label.csv' is in the dataset directory.",                    UserWarning
            )
            return
        
        for path in self.tweet_paths:
            if not os.path.exists(path):
                warn(
                    f"Dataset 'Twibot20.{path}' file not found. Please make sure every file is donwloaded in the dataset directory." 
                )

    def __iter__(self):
        labels = {}
        print("--> Loading labels...")
        with open(self.labels_path, newline="", encoding="utf-8") as f:
            label_data = csv.DictReader(f, delimiter=",")
            for item in label_data:
                user_id = normalize_user_id(item["id"])
                if hash_split_multi(user_id) != self.mode:
                    continue
                labels[normalize_user_id(item["id"])] = item["label"]
        print(f"--> Loaded {len(labels)} labels for split '{self.mode}'")
        print("--> Loading users (This will take a while via ijson)...")
        users = {}
        user_count = 0
        with open(self.user_data_path, "rb") as f:
            for row in ijson.items(f, "item"):
                user_id = normalize_user_id(row["id"])
                if hash_split_multi(user_id) != self.mode:
                    continue
                users[user_id] = row
                user_count += 1
                if user_count % 50000 == 0:
                    #print(f"    Processed {user_count} users...")
                    print(f"\r--> Successfully cached {len(users)} users in memory.", end="", flush=True)

               # 1. Create a temporary database file on your disk
        db_path = os.path.join(os.path.dirname(self.user_data_path), "temp_tweets_cache.db")
        if os.path.exists(db_path):
            os.remove(db_path)

        print("--> Initializing disk cache database...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS tweets (user_id TEXT, raw_json TEXT)")
        conn.commit()

        # 2. Stream lines from files and write to Disk in high-speed batches
        batch = []
        batch_size = 50000  



        for path in self.tweet_paths:
            if not os.path.exists(path):
                continue

            print(f"--> Streaming tweets safely via ijson from: {os.path.basename(path)}")
            tweet_count = 0
            matched_count = 0
            
            # Open in binary mode ("rb") because ijson requires bytes
            with open(path, "rb") as f:
                parser = ijson.items(f, "item")
                try:
                    #tweet_dict = next(parser)
                    for i, tweet_dict in enumerate(parser):
                        tweet_count += 1
                        # Reached the end of the JSON array normally
                        if tweet_count % 100000 == 0:
                            print(f"\r[Streaming] Processed {tweet_count:,} tweets... Matched: {matched_count:,}", end="", flush=True)

                        raw_author_id = tweet_dict.get("author_id", 0)
                        user_id = normalize_user_id(raw_author_id)

                        if user_id in users and user_id in labels:
                            try:
                                # FIX: Use our custom handler to safely serialize Decimal values
                                # create smaller dictionary of only relevant information
                                reduced_dict = {"text": tweet_dict.get("text", "")}
                                raw_json_str = json.dumps(reduced_dict, default=json_decimal_handler)
                                batch.append((user_id, raw_json_str))
                                matched_count += 1
                            except Exception as e:
                                print(f"⚠️ Warning: Failed to serialize tweet for user {user_id}: {e}")
                                continue

                            if len(batch) >= batch_size:
                                cursor.executemany("INSERT INTO tweets VALUES (?, ?)", batch)
                                conn.commit()
                                batch = []
                except StopIteration:
                        break

        if batch:
            cursor.executemany("INSERT INTO tweets VALUES (?, ?)", batch)
            conn.commit()  
                    
    

        # 4. Pull tweets per user sequentially and yield complete samples
        print(f"--> Packaging and yielding complete samples for {len(users)} users...")
        for user_id in users:
            if user_id not in labels:
                continue

            cursor.execute("SELECT raw_json FROM tweets WHERE user_id = ?", (user_id,))
            rows = cursor.fetchall()

            user_tweets = [TweetData.from_row(json.loads(r[0])) for r in rows]

            yield Sample(
                tweet_data=user_tweets,
                user_data=UserData.from_row(users[user_id]),
                label=self.label_mapping.get(labels[user_id],"unknown"),
            )

        # 5. Clean up the database file completely
        conn.close()
        try:
            os.remove(db_path)
            print("--> Temporary disk cache cleaned up successfully.")
        except Exception as e:
            print(f"⚠️ Note: Could not auto-delete temporary database file: {e}")

            

    
    

        