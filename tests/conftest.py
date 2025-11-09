import os
import shutil


def pytest_sessionfinish(session):
    """
    Called after the entire test session finishes.
    """
    test_file_path = "my_test_file.txt"  # The file you want to clean up

    if not session.testsfailed:
        print(f"\nAll tests passed. Cleaning up {test_file_path}")
        """Clean up output directory after tests."""
        output_dir = f"{os.path.dirname(__file__)}/output/"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
