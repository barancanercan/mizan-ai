"""
Turkish Government Intelligence Hub - Exceptions
Custom exception classes for the project.
"""


class IntelligenceHubError(Exception):
    """Base exception for all project errors."""

    def __init__(self, message: str = "An error occurred in Intelligence Hub"):
        self.message = message
        super().__init__(self.message)


class VectorDBError(IntelligenceHubError):
    """Exception raised for vector database errors."""

    def __init__(self, message: str = "Vector database error occurred"):
        super().__init__(message)


class VectorDBNotFoundError(VectorDBError):
    """Exception raised when the vector database does not exist."""

    def __init__(self, db_path: str = "Unknown"):
        message = f"Vector database not found at: {db_path}"
        super().__init__(message)
        self.db_path = db_path


class VectorDBLoadError(VectorDBError):
    """Exception raised when the vector database fails to load."""

    def __init__(self, db_path: str = "Unknown", reason: str = "Unknown error"):
        message = f"Failed to load vector database at {db_path}: {reason}"
        super().__init__(message)
        self.db_path = db_path
        self.reason = reason


class LLMError(IntelligenceHubError):
    """Exception raised for LLM-related errors."""

    def __init__(self, message: str = "LLM error occurred"):
        super().__init__(message)


class LLMConnectionError(LLMError):
    """Exception raised for LLM connection failures."""

    def __init__(self, reason: str = "Connection failed"):
        message = f"LLM connection error: {reason}"
        super().__init__(message)
        self.reason = reason


class LLMResponseError(LLMError):
    """Exception raised for invalid LLM responses."""

    def __init__(self, reason: str = "Invalid response"):
        message = f"LLM response error: {reason}"
        super().__init__(message)
        self.reason = reason


class LLMNotAvailableError(LLMError):
    """Exception raised when no LLM is available."""

    def __init__(self, model: str = "Unknown"):
        message = f"No LLM available: {model} is not accessible"
        super().__init__(message)
        self.model = model


class DataError(IntelligenceHubError):
    """Exception raised for data processing errors."""

    def __init__(self, message: str = "Data processing error occurred"):
        super().__init__(message)


class PDFFileNotFoundError(DataError):
    """Exception raised when a PDF file is not found."""

    def __init__(self, file_path: str = "Unknown"):
        message = f"PDF file not found: {file_path}"
        super().__init__(message)
        self.file_path = file_path


class PDFLoadError(DataError):
    """Exception raised when a PDF file fails to load."""

    def __init__(self, file_path: str = "Unknown", reason: str = "Unknown error"):
        message = f"Failed to load PDF at {file_path}: {reason}"
        super().__init__(message)
        self.file_path = file_path
        self.reason = reason


class PartyNotFoundError(DataError):
    """Exception raised when a party is not found or unknown."""

    def __init__(self, party: str = "Unknown"):
        message = f"Unknown party: {party}"
        super().__init__(message)
        self.party = party


class ConfigurationError(IntelligenceHubError):
    """Exception raised for configuration-related issues."""

    def __init__(self, message: str = "Configuration error occurred"):
        super().__init__(message)
