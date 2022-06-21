import os
import re
from typing import Dict


def get_cpc_texts(cpc_scheme_xml_dir: str, cpc_title_list_dir: str) -> Dict[str, str]:
    """
    Collects a context text for each context code from raw files

    Args:
        cpc_scheme_xml_dir (str): directory where cpc_scheme_xml files are saved
        cpc_title_list_dir (str): directory where cpc_title_list files are saved

    Returns:
        Dict[str, str]: a dictionary with key as context code and value as context text
    """

    contexts = []
    pattern = "[A-Z]\d+"
    for file_name in os.listdir(cpc_scheme_xml_dir):
        result = re.findall(pattern, file_name)
        if result:
            contexts.append(result)
    contexts = sorted(set(sum(contexts, [])))
    results = {}
    for cpc in ["A", "B", "C", "D", "E", "F", "G", "H", "Y"]:
        with open(f"{cpc_title_list_dir}/cpc-section-{cpc}_20220201.txt") as f:
            s = f.read()
        pattern = f"{cpc}\t\t.+"
        result = re.findall(pattern, s)
        cpc_result = result[0].lstrip(pattern)
        for context in [c for c in contexts if c[0] == cpc]:
            pattern = f"{context}\t\t.+"
            result = re.findall(pattern, s)
            results[context] = cpc_result + ". " + result[0].lstrip(pattern)
    return results
