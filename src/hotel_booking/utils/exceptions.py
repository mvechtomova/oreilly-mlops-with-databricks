class MissingDeploymentTagError(Exception):
    """Raised when the required deployment tag is missing."""
    pass

class DeploymentNotApprovedError(Exception):
    """Raised when the model is tagged but not approved for deployment."""
    pass
