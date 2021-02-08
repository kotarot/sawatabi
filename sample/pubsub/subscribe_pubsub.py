#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2021 Kotaro Terada
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

from google.cloud import pubsub_v1

"""
This script subscribes messages from GCP Pub/Sub for debugging.

Sample Usage:
$ GOOGLE_APPLICATION_CREDENTIALS="./gcp-key.json" python sample/pubsub/subscribe_pubsub.py \
    --project=your-project\
    --subscription=your-subscription
"""


def main() -> None:
    parser = argparse.ArgumentParser()

    # fmt: off

    # Pub/Sub options
    parser.add_argument(
        "--project",
        dest="project",
        required=True,
        help="Google Cloud Platform project name.")
    parser.add_argument(
        "--subscription",
        dest="subscription",
        required=True,
        help="Google Cloud Pub/Sub subscription name to subscribe messages from.")

    # fmt: on

    args = parser.parse_args()

    client = pubsub_v1.SubscriberClient()
    subscription_path = client.subscription_path(args.project, args.subscription)

    def callback(message: pubsub_v1.subscriber.message.Message) -> None:
        dt_now = datetime.datetime.now()
        print(f"[{dt_now}] Received message from '{subscription_path}': {message.data.decode('utf-8')}")
        message.ack()

    streaming_pull_future = client.subscribe(subscription_path, callback=callback)

    with client:
        try:
            streaming_pull_future.result(timeout=None)
        except TimeoutError:
            streaming_pull_future.cancel()


if __name__ == "__main__":
    main()
