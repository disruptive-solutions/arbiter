import pefile
from hashlib import sha256

from pathlib import Path
from sqlalchemy.exc import DatabaseError

from models import SampleData, init_db


# Helper functions to get all the info we need using pefile

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


def main(dir_path: Path, db_path: Path):
    """
    Iterate over `dir_path`, get information from each file, add to database

    :param dir_path: The directory to iterate
    :param db_path: The path of the database to create/update
    """
    db_session = init_db(db_path)
    directory = dir_path.expanduser()

    # Initialize some counters for a sanity check
    num_files = 0
    num_pe = 0
    num_dupes = 0
    dos_files = []

    for file in directory.iterdir():
        if not file.is_file():  # Skip over directories
            continue

        num_files += 1

        if not file.read_bytes()[:2] == b"MZ":  # Skip over non-PE files
            continue

        try:
            # So that we can use pefile
            pe_file = pefile.PE(file)

            # Create a database entry
            sample = SampleData(sha256=sha256(file.read_bytes()).hexdigest(),
                                debug_size=get_debug_size(pe_file),
                                image_version=get_image_version(pe_file),
                                import_rva=get_import_rva(pe_file),
                                export_size=get_export_size(pe_file),
                                resource_size=get_resource_size(pe_file),
                                num_sections=get_num_sections(pe_file),
                                virtual_size_2=get_virtual_size_2(pe_file))

            # Add the database entry to the database
            db_session.add(sample)
            db_session.commit()
            print(f"Added {str(file)} to the database")
            num_pe += 1

        except pefile.PEFormatError:
            dos_files.append(str(file))
            continue

        except DatabaseError:
            # Don't add if its SHA256 is already in there
            db_session.rollback()
            print(f"Failed to add {str(file)} to the database")
            num_dupes += 1
            continue

        except (TypeError, AttributeError, IndexError):
            continue

    print(f"\nDuplicates:{str(num_dupes):>35}")
    print(f"b'MZ' files pefile can't handle (DOS?): {str(len(dos_files)):>6}")
    print(f"PE files added to the database:{str(num_pe):>15}")
    print(f"Total files seen:{str(num_files):>29}")
    print("\nPossible DOS-only files:")
    print(dos_files)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path, help="The directory where all the malware is stored")
    p.add_argument('database', type=Path, help="The path to a SQLite database to create/update")

    args = p.parse_args()
    main(args.directory, args.database)
