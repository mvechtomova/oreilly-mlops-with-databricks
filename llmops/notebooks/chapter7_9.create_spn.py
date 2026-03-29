# Databricks notebook source
from databricks.sdk import WorkspaceClient
from requests.auth import HTTPBasicAuth

w = WorkspaceClient()

client_id = "your client id"
client_secret = "your client secret"
account_id = "your account id"

w.secrets.create_scope(scope="admin")
w.secrets.put_secret(scope="admin", key="client_id", string_value=client_id)
w.secrets.put_secret(scope="admin", key="client_secret", string_value=client_secret)
w.secrets.put_secret(scope="admin", key="account_id", string_value=account_id)


# COMMAND ----------
import requests
from databricks.sdk import WorkspaceClient
from requests.auth import HTTPBasicAuth
import urllib

w = WorkspaceClient()

# Admin credentials from secret scope
admin_client_id = dbutils.secrets.get("admin", "client_id")
admin_client_secret = dbutils.secrets.get("admin", "client_secret")
account_id = dbutils.secrets.get("admin", "account_id")

account_host = "https://accounts.cloud.databricks.com"
instance_name = "arxiv-agent-instance"

# Get account-level token
token = requests.post(
    f"{account_host}/oidc/accounts/{account_id}/v1/token",
    auth=HTTPBasicAuth(admin_client_id, admin_client_secret),
    data={"grant_type": "client_credentials", "scope": "all-apis"}
).json()["access_token"]

# Step 1: Create service principal + OAuth secret
sp = w.service_principals.create(display_name="lakebase-sp-arxiv")
secret_resp = requests.post(
    f"{account_host}/api/2.0/accounts/{account_id}/servicePrincipals/{sp.id}/credentials/secrets",
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
)
secret_resp.raise_for_status()
client_id = sp.application_id
client_secret = secret_resp.json()["secret"]

# COMMAND ----------
# Step 2: Store credentials in a secret scope
scope_name = "arxiv-agent-scope"
try:
    w.secrets.create_scope(scope=scope_name)
except Exception:
    pass  # scope already exists
w.secrets.put_secret(scope=scope_name, key="client_id", string_value=client_id)
w.secrets.put_secret(scope=scope_name, key="client_secret", string_value=client_secret)

# COMMAND ----------
# Step 3: Add SPN role to project
from databricks.sdk.service.postgres import (
    PostgresAPI, Role, RoleAuthMethod, RoleIdentityType, RoleRoleSpec,
)
import psycopg

project_id = "arxiv-agent-lakebase"
w = WorkspaceClient()
pg_api = PostgresAPI(w.api_client)

project = pg_api.get_project(name=f"projects/{project_id}")
default_branch = next(iter(pg_api.list_branches(parent=project.name)))
branch_parent = default_branch.name

pg_api.create_role(
    parent=branch_parent,
    role=Role(
        spec=RoleRoleSpec(
            identity_type=RoleIdentityType.SERVICE_PRINCIPAL,
            auth_method=RoleAuthMethod.LAKEBASE_OAUTH_V1,
            postgres_role=client_id,
        )
    ),
    role_id="arxiv-agent-spn",
).wait()

# COMMAND ----------
# Step 4: Postgres role SQL 
endpoint = next(iter(pg_api.list_endpoints(parent=branch_parent)))
host = endpoint.status.hosts.host
pg_credential = pg_api.generate_database_credential(endpoint=endpoint.name)

user = w.current_user.me()
username = urllib.parse.quote_plus(user.user_name)

conn_string = (
    f"postgresql://{username}:{pg_credential.token}@{host}:5432/"
    "databricks_postgres?sslmode=require"
)

with psycopg.connect(conn_string) as conn:
    conn.execute(f"""
        GRANT USAGE ON SCHEMA public TO "{client_id}";
    """)
    conn.execute(f"""
        GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE session_messages TO "{client_id}";
    """)
    conn.execute(f"""
        GRANT USAGE, SELECT ON SEQUENCE session_messages_id_seq TO "{client_id}";
    """)
