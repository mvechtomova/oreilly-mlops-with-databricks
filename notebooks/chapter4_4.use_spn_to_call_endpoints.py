import os

import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

load_dotenv()


def get_token():
    response = requests.post(
        f"https://{os.environ['DBR_HOST']}/oidc/v1/token",
        auth=HTTPBasicAuth(
            os.environ["DATABRICKS_CLIENT_ID"],
            os.environ["DATABRICKS_CLIENT_SECRET"]
        ),
        data={
            'grant_type': 'client_credentials',
            'scope': 'all-apis'
        }
    )
    return response.json()["access_token"]

token = get_token()
