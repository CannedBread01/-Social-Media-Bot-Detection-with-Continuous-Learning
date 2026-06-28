import shutil
import csv
import ijson
import os
from warnings import warn
import lmdb
from dataset_utils.constants import UserData, TweetData, Sample, normalize_user_id
from torch.utils.data import IterableDataset


class Twibot22Improved(IterableDataset):
    """
    Dataset for the Twibot-22. The downloaded *.json files or a directory,named "Twibot22", containing them, need to be in the directory /datasets.
    IMPORTANT: IF YOU CAN DOWNLOAD THEM DIRECTLY INTO THE TWIBOT22 FOLDER.
    """

    def __init__(self, mode: str, train_split: float = 0.8, dev_split: float = 0.1, root: str | None = None,
                 label_mapping=["bot", "human"], sub_sample_size = -1):
        """
        :param root: the filepath of the dataset directory in which the dataset is stored.
        :param mode: Dataset split to use ("train", "dev", or "test").
        :param train_split: Fraction of users assigned to the training set (e.g. 0.8 = 80%).
        :param dev_split: Fraction of users assigned to the validation set (e.g. 0.1 = 10
        :param label_mapping: List of labels to use for provided samples. Format: `['bot', 'human', 'unlabelled']`
        :param sub_sample_size: (Default: -1) the upper limit of the amount samples the iterator should provide. Fewer samples are possible if the dataset doesn't have enough samples.
        A value of `-1` provides the default amount of samples.
        """
        if ijson.backend_name == 'python': print("WARNING: ijson is running in ultra-slow pure-Python mode! Use/Install libyajl2 instead for faster processing.")

        # __init_ does the file handling
        if root == None: root = "datasets"

        self.sub_sample_size = sub_sample_size
        # remap split label to work with provided dataset
        mode_map = {
            "train": "train",
            "dev": "val",
            "test": "test"
        }
        self.mode = mode_map[mode]

        # mapping for custom labels
        self.label_mapping = {
            'bot': label_mapping[0],
            'human': label_mapping[1],
        }

        assert train_split + dev_split < 1.0

        # creates subdirectory "Twibot22" if it doesn't exist
        if not os.path.exists(os.path.join(root, "Twibot22")):
            os.makedirs(os.path.join(root, "Twibot22"), exist_ok=True)

        # moves all json files into the "Twibot22" subdirectory
        for file in os.listdir(root):
            if file.endswith(".json") | file.endswith(".csv"):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(root, "Twibot22", file)
                shutil.move(src_path, dst_path)

        # defines the paths to the user data, labels and tweets
        self.tweet_database_path = os.path.join(root, "Twibot22", "tweetDB")
        self.split_path = os.path.join(root, "Twibot22", "split.csv")
        self.user_data_path = os.path.join(root, "Twibot22", "user.json")
        self.labels_path = os.path.join(root, "Twibot22", "label.csv")
        self.tweet_paths = [
            os.path.join(root, "Twibot22", f"tweet_{i}.json")
            for i in range(9)
        ]

        # validation of the expected paths
        if not os.path.exists(self.user_data_path): # user data
            warn("Dataset 'Twibo22.user.json' file not found. Please make sure 'user.json' is in the dataset directory.",UserWarning)
            return

        if not os.path.exists(self.labels_path): # label data
            warn("Dataset 'Twibot22.label.tsv' file not found. Please make sure 'label.csv' is in the dataset directory.",UserWarning)
            return

        if not os.path.exists(self.split_path): # split data
            warn("Dataset 'Twibot22.split.csv' file not found. Please make sure 'split.csv' is in the dataset directory.",UserWarning)
            return


        for path in self.tweet_paths:
            if not os.path.exists(path):
                warn(f"Dataset 'Twibot20.{path}' file not found. Please make sure every file is downloaded in the dataset directory.")

    def stream_tweet_data(self,json_path: str):
        """
        goes through the provided json and returns an iterator that yields tweet information `(author_id, text)`
        :param json_path: path to the json file
        :return: yields tweet information `(author_id, text)`
        """

        # ijson doesn't parse the json into dictionary objects, but instead goes through file in a moving window
        with open(json_path, 'rb') as f:
            parser = ijson.parse(f)

            author_id = None
            text = None
            parent_prefix = None # needed to ensure that don't falsely extract data from deeper nested objects

            for prefix, event, value in parser:
                # extract author id from tweet
                if prefix.endswith('.author_id'):
                    author_id = value
                    # Dynamically find the parent object prefix (e.g., 'item' or 'tweets.item')
                    if parent_prefix is None: parent_prefix = prefix.rsplit('.', 1)[0]
                # extract tweet text
                elif prefix.endswith('.text'):
                    text = value
                    if parent_prefix is None: parent_prefix = prefix.rsplit('.', 1)[0]

                # yield once the end of a json object (tweet object) is detected
                elif event == 'end_map' and prefix == parent_prefix:
                    if author_id is not None and text is not None:
                        yield author_id, text

                    # Reset for next iteration
                    author_id = None
                    text = None

    def prepare_dataset(self):
        """
        creates a database for storing the tweet text in more efficient manner.
        Call this function once to create the database files, before iterating on the dataset.
        If the previous database is incomplete or outdated, delete the directory of the database and rerun the function.
        """
        # check if database already exists, if it does, abort inorder to not override it
        if not os.path.exists(self.tweet_database_path): # split data
            warn(f"Already existing Tweet database found. If database should be recreated, then delete the directory at {self.tweet_database_path}",UserWarning)
            return

        # setup database
        env = lmdb.open(self.tweet_database_path, map_size=1099511627776, max_dbs=1)
        tweet_db = env.open_db(b'tweets', dupsort=True)
        txn = env.begin(write=True, db=tweet_db)

        batch = []
        batch_size = 50000

        for path in self.tweet_paths:
            # skip non existing files
            if not os.path.exists(path): continue
            with txn.cursor() as cursor:

                print(f"Streaming tweets safely via ijson from: {os.path.basename(path)}")
                tweet_count = 0

                # go through the tweet files and extract the tweets
                for author_id, text in self.stream_tweet_data(path):
                    tweet_count += 1
                    if tweet_count % 100000 == 0:
                        print(f"[Streaming] Processed {tweet_count:,} tweets...")

                    user_id = normalize_user_id(author_id)
                    # add to batch
                    batch.append(
                        (user_id.encode("utf-8"),
                        text.encode("utf-8")[:511]) # byte length is limited due to database constraints (doesn't affect most tweets)
                    )

                    # add batch to database
                    if len(batch) >= batch_size:
                        batch.sort(key=lambda x: x[0]) # sorting by keys for faster addition in database
                        cursor.putmulti(batch) # adding to database
                        batch = [] # resetting for next batch
                        txn.commit() # commits changes to database
                        txn = env.begin(write=True, db=tweet_db) # start new transaction, as commit finishes the current one
                        cursor = txn.cursor()
                # process the last remaining tweets
                if batch:
                    batch.sort(key=lambda x: x[0])
                    cursor.putmulti(batch)
                    batch = []
                    txn.commit()
        env.close()


    def __iter__(self):
        # get users IDs for the relevant split
        user_list = []
        with open(self.split_path, newline="", encoding="utf-8") as f:
            split_data = csv.DictReader(f, delimiter=",")
            for item in split_data:
                # skip not relevant entries
                if item["split"] != self.mode: continue
                # add user to list
                user_list.append(normalize_user_id(item["id"]))

        # get labels for relevant users
        labels = {}
        print("--> Loading labels...")
        with open(self.labels_path, newline="", encoding="utf-8") as f:
            label_data = csv.DictReader(f, delimiter=",")
            for item in label_data:
                labels[normalize_user_id(item["id"])] = item["label"]

        # get relevant users
        users = {}
        user_count = 0
        with open(self.user_data_path, "rb") as f:
            for row in ijson.items(f, "item"):
                user_id = normalize_user_id(row["id"])
                users[user_id] = row
                user_count += 1
                if user_count % 50000 == 0:
                    print(f"\r--> Successfully cached {len(users)} users in memory.", end="", flush=True)

        # prepare access to database for tweets
        env = lmdb.open(self.tweet_database_path, max_dbs=1, readonly=True)
        tweet_db = env.open_db(b'tweets', dupsort=True)

        with env.begin(db=tweet_db) as txn:
            with txn.cursor() as cursor:

                # loop over users
                iteration = -self.sub_sample_size
                for user_id in user_list:
                    # get tweets from database
                    tweets = []

                    if cursor.set_key(str(user_id).strip().encode("utf-8")):
                        tweets.append(cursor.value().decode("utf-8", errors="ignore"))
                        while cursor.next_dup():
                            tweets.append(cursor.value().decode("utf-8", errors="ignore"))


                    user_data = UserData.from_row(users.get(user_id))
                    label = self.label_mapping.get(labels[user_id], "unknown"),
                    tweet_data = [TweetData.from_row({"text":tweet}) for tweet in tweets]
                    # yield sample
                    yield Sample(
                        tweet_data=tweet_data,
                        user_data=user_data,
                        label=label,
                    )
                    # count and stop early if only a sub sample is requested
                    if self.sub_sample_size > 0:
                        iteration += 1
                        if iteration >= 0: break
        env.close()