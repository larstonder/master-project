"""
This module provides a context manager to temporarily modify sys.path
and optionally change the current working directory (cwd).
"""
from contextlib import contextmanager
import sys
import os

@contextmanager
def local_import_context(path: str, change_cwd: bool = False):
    """
    Context manager to temporarily modify sys.path and optionally change cwd.
    
    :param path: The path to temporarily add to sys.path and/or cwd.
    :param change_cwd: Whether to change working directory.
    """
    print("Entered local import context")
    print(f"Adding {path} to sys.path")
    print(f"Current sys.path: {sys.path}")

    path = os.path.abspath(path)
    old_sys_path = list(sys.path)
    old_cwd = os.getcwd()

    if path not in sys.path:
        sys.path.insert(0, path)
    if change_cwd:
        os.chdir(path)

    try:
        yield
    finally:
        sys.path = old_sys_path
        if change_cwd:
            os.chdir(old_cwd)

@contextmanager
def use_path(path: str, change_cwd: bool = True):
    """
    Context manager for temporarily adding a path to sys.path and changing working directory.
    This ensures that all operations inside the context use the new path,
    and everything is restored properly afterwards.
    
    Usage:
    with use_path("drivestudio"):
        # code that needs the drivestudio path
        # paths are relative to drivestudio directory
    
    :param path: The path to add to sys.path and/or change to
    :param change_cwd: Whether to change working directory
    """
    path = os.path.abspath(path)
    old_sys_path = list(sys.path)
    old_cwd = os.getcwd()

    # Add to sys.path if not already there
    if path not in sys.path:
        sys.path.insert(0, path)
        print(f"Added {path} to sys.path")

    # Change working directory if requested
    if change_cwd:
        os.chdir(path)
        print(f"Changed working directory to {path}")

    try:
        yield
    finally:
        # Restore original settings
        sys.path = old_sys_path
        print("Restored original sys.path")
        
        if change_cwd:
            os.chdir(old_cwd)
            print(f"Restored working directory to {old_cwd}")