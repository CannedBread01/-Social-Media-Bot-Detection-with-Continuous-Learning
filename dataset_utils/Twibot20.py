import json
import ijson
from zipfile import ZipFile
from warnings import warn
from torch.utils.data import IterableDataset
try:
    from dataset_utils.constants import UserData, TweetData, Sample, normalize_user_id,_safe_dict, clean_text
except ImportError:
    from dataset_utils.constants import UserData, TweetData, Sample, normalize_user_id, _safe_dict, clean_text
import os
from dataset_utils.splitting import hash_split_multi
class Twibot20(IterableDataset):
    """
    Dataset for the Twibot-20. The downloaded *.json files or a directory containing them,named "Twibot20", need to be in the directory /datasets.
    """
    def __init__(self, mode :str, train_split: float = 0.8, dev_split: float = 0.1, root : str |None = None, label_mapping = ["bot","human"], sub_sample_size = -1):
        """
        :param root(OPTIONAL): the filepath of the dataset directory in which the dataset is stored, if none is given "datasets"
        :param mode: Dataset split to use ("train", "dev", or "test").
        :param train_split: Fraction of users assigned to the training set (e.g. 0.8 = 80%).
        :param dev_split: Fraction of users assigned to the validation set (e.g. 0.1 = 10)
        :param label_mapping: List of labels to use for provided samples. Format: `['bot', 'human', 'unlabelled']`
        :param sub_sample_size: (Default: -1) the upper limit of the amount samples the iterator should provide. Fewer samples are possible if the dataset doesn't have enough samples.
        """
        if root is None:
            root = "datasets"

        self.sub_sample_size = sub_sample_size
        self.mode = mode
        self.label_mapping = label_mapping

        assert train_split + dev_split < 1.0

        if not os.path.exists(os.path.join(root, "Twibot-20")):
            default_path = os.path.join(root, "data.zip")
            if not os.path.exists(default_path):
                warn(
                "Dataset 'Twibot20' file not found. Please make sure 'Twibot-20.zip' is in the dataset directory.",                    UserWarning
                )
                return 

            with ZipFile(default_path) as zipObj:
                zipObj.extractall(os.path.join(root))
            os.remove(default_path)  # removes the zip file after extraction to save space
            print("Finished 'Twibot-20' extraction and removed the zip file")

        self.data_path_train = os.path.join(root, "Twibot-20", "train.json")
        self.data_path_dev = os.path.join(root, "Twibot-20", "dev.json")
        self.data_path_test = os.path.join(root,"Twibot-20", "test.json")

        if not os.path.exists(self.data_path_test):
            warn(
                "Dataset Twibot20's 'test.json' file not found. Please make sure 'test.json' is in the dataset directory.",
                UserWarning,
            )

        if not os.path.exists(self.data_path_train):
            warn(
                "Dataset Twibot20's 'train.json' file not found. Please make sure 'train.json' is in the dataset directory.",
                UserWarning,
            )
            return

        if not os.path.exists(self.data_path_dev):
            warn(
                "Dataset Twibot20's 'dev.json' file not found. Please make sure 'dev.json' is in the dataset directory.",
                UserWarning,
            )
            return
        
        #helps looking up all of them and in order
        self.files = [
        self.data_path_test,
        self.data_path_train,
        self.data_path_dev,
    ]
        

    def __iter__(self):
        iteration = -self.sub_sample_size
        for path in self.files:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                current_tweets = []
                current_user_id = None
                #for the flush of the last user
                old_user = None
                previous_label = None
                for row in data:
                    
                    # creates a user
                    user_id = normalize_user_id(row.get("ID"))
                    split = hash_split_multi(user_id)
                    if split != self.mode:
                        continue
                    profile = _safe_dict(row.get("profile"))
                    user = UserData.from_row({
                        "id": user_id,
                        "profile": profile
                    })
    

                    #creates the bot_label
                    bot_label = row.get("label")
                    if bot_label == "1":
                        bot_label = self.label_mapping[0]
                    elif bot_label == "0":
                        bot_label = self.label_mapping[1]
                    else:
                        continue

                    if previous_label is None:
                        previous_label = bot_label

                    if old_user is None:
                        old_user = user

                    if current_user_id is None:
                        current_user_id = user_id

                    #creates the tweet list
                    # Has the User changed?
                    # yes -> reininitilize the tweet_list, as well as the current_user, 
                    # and yield the tweets of the old list
                    # no --> append the tweet list
                    if current_user_id != user_id:

                        yield Sample(
                                tweet_data=current_tweets,
                                user_data=old_user,
                                label=previous_label,
                            )
                        # count and stop early if only a sub sample is requested
                        if self.sub_sample_size > 0:
                            iteration += 1
                            if iteration >= 0: break

                        current_tweets = []
                        current_user_id = user_id
                        old_user = user
                        previous_label = bot_label
                        
                    tweet_id = 0
                    tweets = row.get("tweet",[])
                    if tweets is None:
                        continue
                    for tweet in tweets:
                        tweet_id += 1
                        tweet = str(tweet).strip()
                        if not tweet:
                            continue

                        tweet_row = {
                            "id": tweet_id,
                            "text": tweet,
                            "user_id": user_id,
                            "created_at": None,
                            "public_metrics": {},
                            "entities": {},
                        }
                        
                        current_tweets.append(
                            TweetData.from_row(tweet_row)
                        )

                #flush out the last_user
                if current_tweets:
                    if current_user_id == None:
                        continue
                        
                    yield Sample(
                        tweet_data=current_tweets,
                        user_data=old_user,
                        label=previous_label,
                    )
                    # count and stop early if only a sub sample is requested
                    if self.sub_sample_size > 0:
                        iteration += 1
                        if iteration >= 0: break

                    

            