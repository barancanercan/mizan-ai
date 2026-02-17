"""
Turkish Government Intelligence Hub - Party Utility Functions
Parti yardımcı fonksiyonları
"""

from typing import List


def normalize_party_name(party: str) -> str:
    """
    Normalize party name with proper Turkish character handling.

    Args:
        party: Party name to normalize

    Returns:
        str: Normalized party name
    """
    party_upper = party.upper()
    
    if party_upper in ("IYI", "İYİ"):
        return "İYİ"
    
    return party_upper


def normalize_parties_list(parties: List[str]) -> List[str]:
    """
    Normalize a list of party names and return sorted unique list.

    Args:
        parties: List of party names

    Returns:
        List[str]: Sorted list of unique normalized party names
    """
    normalized = {normalize_party_name(party) for party in parties}
    return sorted(normalized)


def get_party_display_name(party: str) -> str:
    """
    Get short display name for a party.

    Args:
        party: Party name

    Returns:
        str: Short display name for the party
    """
    return normalize_party_name(party)


def validate_party_code(party: str, valid_parties: List[str]) -> bool:
    """
    Validate if party code is valid after normalization.

    Args:
        party: Party code to validate
        valid_parties: List of valid party codes

    Returns:
        bool: True if party is valid after normalization
    """
    normalized = normalize_party_name(party)
    return normalized in valid_parties
