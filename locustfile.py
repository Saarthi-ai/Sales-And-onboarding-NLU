import random
import pandas as pd
from locust import HttpUser, task, between


data = pd.read_csv('data.csv', dtype_backend='pyarrow', engine='pyarrow')
utterances = data['text'].tolist()
del data


class PhoneCallUser(HttpUser):
    wait_time = between(2, 4)

    @task
    def utter(self):
        with self.client.request('POST', '/predict', json={'text': random.choice(utterances)}, catch_response=True) as response:
            if response.elapsed.total_seconds() > 0.7:
                response.failure('Response took too long')
