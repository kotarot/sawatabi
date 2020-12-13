# Copyright 2020 Kotaro Terada
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import datetime
import random
import string
import time

from google.cloud import pubsub_v1
#from google.oauth2 import service_account


"""
This script publishes test messages to GCP Pub/Sub for debug.

Sample Usage:
$ GOOGLE_APPLICATION_CREDENTIALS="sample/trial/gcp-key.json" python sample/trial/publish_pubsub.py --project=your-project --topic=your-topic
$ GOOGLE_APPLICATION_CREDENTIALS="sample/trial/gcp-key.json" python sample/trial/publish_pubsub.py --project=your-project --topic=your-topic --interval=10.0 --random-text
$ GOOGLE_APPLICATION_CREDENTIALS="sample/trial/gcp-key.json" python sample/trial/publish_pubsub.py --project=your-project --topic=your-topic --interval=30.0 --random-number
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        dest="project",
        required=True,
        help="Google Cloud Pub/Sub project name.")
    parser.add_argument(
        "--topic",
        dest="topic",
        required=True,
        help="Google Cloud Pub/Sub topic name to publish messages to.")
    parser.add_argument(
        "--random-text",
        dest="random_text",
        action="store_true",
        help="If true, a message will be a random text string.")
    parser.add_argument(
        "--random-number",
        dest="random_number",
        action="store_true",
        help="If true, a message will be a random number.")
    parser.add_argument(
        "--incremental-text",
        dest="incremental_text",
        action="store_true",
        help="If true, a message will be a text string whose length increments.")
    parser.add_argument(
        "--incremental-number",
        dest="incremental_number",
        action="store_true",
        help="If true, a message will be a incremental number.")
    parser.add_argument(
        "--interval",
        dest="interval",
        type=float,
        default=1.0,
        help="Message interval in second.")
    args = parser.parse_args()

    #client = pubsub_v1.PublisherClient(credentials=service_account.Credentials.from_service_account_file(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]))
    client = pubsub_v1.PublisherClient()
    topic_path = client.topic_path(args.project, args.topic)

    i = 1
    while True:
        dt_now = datetime.datetime.now()
        if args.random_text:
            words = []
            for _ in range(random.randint(1, 10)):
                word = "".join([random.choice(string.ascii_letters + string.digits) for i in range(random.randint(1, 10))])
                words.append(word)
            message = " ".join(words) + "."
        elif args.random_number:
            n = random.randint(1, 99)
            message = str(n)
        elif args.incremental_text:
            message = "".join([random.choice(string.ascii_letters + string.digits) for i in range(i)])
        elif args.incremental_number:
            message = str(i)
        else:
            message = f"This is a test message at {dt_now}."

        client.publish(topic_path, message.encode("utf-8"))
        print(f"[{dt_now}] Published a message to '{topic_path}': {message}")

        i += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
