class DeploymentNotApprovedError(Exception):
    """Raised when a model version is not approved for deployment."""

    pass


class MissingDeploymentTagError(Exception):
    """Raised when a required deployment tag is missing."""

    pass
