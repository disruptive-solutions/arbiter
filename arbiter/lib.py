from hashlib import sha256

import pefile

from pathlib import Path
from typing import Dict, List


def get_debug_size(pe: pefile.PE) -> int:
    # 6 is DIRECTORY_ENTRY_DEBUG
    return pe.OPTIONAL_HEADER.DATA_DIRECTORY[6].Size


def get_image_version(pe: pefile.PE) -> int:
    return pe.OPTIONAL_HEADER.MajorImageVersion


def get_import_rva(pe: pefile.PE) -> int:
    # 1 is DIRECTORY_ENTRY_IMPORT
    return pe.OPTIONAL_HEADER.DATA_DIRECTORY[1].VirtualAddress


def get_export_size(pe: pefile.PE) -> int:
    # 0 is DIRECTORY_ENTRY_EXPORT
    return pe.OPTIONAL_HEADER.DATA_DIRECTORY[0].Size


def get_resource_size(pe: pefile.PE) -> int:
    # 2 is DIRECTORY_ENTRY_RESOURCE
    return pe.OPTIONAL_HEADER.DATA_DIRECTORY[2].Size


def get_num_sections(pe: pefile.PE) -> int:
    return pe.FILE_HEADER.NumberOfSections


def get_virtual_size_2(pe: pefile.PE) -> int:
    # 1 to index it to the second section
    # Return 0 if there isn't a second section
    try:
        size = pe.sections[1].Misc_VirtualSize
        return size
    except (AttributeError, TypeError, IndexError):
        return 0


def build_sample(path: Path) -> Dict:
    try:
        pe = pefile.PE(path)
    except pefile.PEFormatError as e:
        raise ValueError(f"{e}")
    else:
        return {'debug_size': get_debug_size(pe),
                'image_version': get_image_version(pe),
                'import_rva': get_import_rva(pe),
                'export_size': get_export_size(pe),
                'resource_size': get_resource_size(pe),
                'num_sections': get_num_sections(pe),
                'virtual_size_2': get_virtual_size_2(pe)}


def build_corpus(paths: List[Path]) -> Dict:
    seen_files = dict()

    for path in paths:
        hash_ = sha256(path.read_bytes()).hexdigest()
        if hash_ in seen_files:
            continue

        try:
            seen_files[hash_] = build_sample(path)
        except ValueError:
            print(f"Unable to process {path.name}")

    return seen_files
