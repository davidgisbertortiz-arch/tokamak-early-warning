#!/usr/bin/env python3
"""
README link and asset auditor.

Parses README.md and checks that all local references (images, links) exist.
Returns non-zero exit code if any references are broken.

Usage:
    python scripts/check_readme_assets.py
    python scripts/check_readme_assets.py --readme README.md
"""

import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Set


def extract_markdown_references(content: str) -> List[Tuple[str, str, int]]:
    """
    Extract all markdown references from content.
    
    Returns list of (reference_type, target, line_number) tuples.
    reference_type is 'image' for ![]() or 'link' for []()
    """
    references = []
    lines = content.split('\n')
    
    # Patterns for markdown links and images
    # Image: ![alt](path) or <img src="path"...>
    # Link: [text](path)
    
    image_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
    link_pattern = re.compile(r'(?<!!)\[([^\]]*)\]\(([^)]+)\)')
    html_img_pattern = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE)
    
    for line_num, line in enumerate(lines, 1):
        # Find markdown images
        for match in image_pattern.finditer(line):
            target = match.group(2).split()[0]  # Handle ![](path "title")
            references.append(('image', target, line_num))
        
        # Find markdown links
        for match in link_pattern.finditer(line):
            target = match.group(2).split()[0]
            references.append(('link', target, line_num))
        
        # Find HTML images
        for match in html_img_pattern.finditer(line):
            target = match.group(1)
            references.append(('image', target, line_num))
    
    return references


def is_local_reference(target: str) -> bool:
    """Check if a reference is local (not external URL or anchor)."""
    # Skip external URLs
    if target.startswith(('http://', 'https://', 'mailto:', 'ftp://')):
        return False
    # Skip anchors
    if target.startswith('#'):
        return False
    # Skip data URIs
    if target.startswith('data:'):
        return False
    return True


def check_references(
    readme_path: Path,
    repo_root: Path
) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
    """
    Check all local references in README.
    
    Returns (missing, found) lists of (type, target, line) tuples.
    """
    content = readme_path.read_text(encoding='utf-8')
    references = extract_markdown_references(content)
    
    missing = []
    found = []
    seen: Set[str] = set()
    
    for ref_type, target, line_num in references:
        if not is_local_reference(target):
            continue
        
        # Skip duplicates
        if target in seen:
            continue
        seen.add(target)
        
        # Clean target (remove query params, anchors in local paths)
        clean_target = target.split('?')[0].split('#')[0]
        
        # Check if file exists relative to repo root
        target_path = repo_root / clean_target
        
        if target_path.exists():
            found.append((ref_type, target, line_num))
        else:
            missing.append((ref_type, target, line_num))
    
    return missing, found


def main():
    parser = argparse.ArgumentParser(
        description="Check README.md for broken local references"
    )
    parser.add_argument(
        "--readme", type=str, default="README.md",
        help="Path to README file (default: README.md)"
    )
    parser.add_argument(
        "--repo-root", type=str, default=None,
        help="Repository root directory (default: parent of README)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show all references, not just broken ones"
    )
    
    args = parser.parse_args()
    
    # Determine paths
    readme_path = Path(args.readme).resolve()
    if not readme_path.exists():
        print(f"❌ README not found: {readme_path}")
        sys.exit(1)
    
    if args.repo_root:
        repo_root = Path(args.repo_root).resolve()
    else:
        # Assume repo root is parent of README or where .git is
        repo_root = readme_path.parent
        git_dir = repo_root / ".git"
        while not git_dir.exists() and repo_root.parent != repo_root:
            repo_root = repo_root.parent
            git_dir = repo_root / ".git"
    
    print(f"📄 Checking: {readme_path.relative_to(repo_root)}")
    print(f"📁 Repo root: {repo_root}")
    print()
    
    missing, found = check_references(readme_path, repo_root)
    
    # Report results
    if args.verbose and found:
        print(f"✅ Found {len(found)} valid local references:")
        for ref_type, target, line_num in found:
            icon = "🖼️" if ref_type == "image" else "🔗"
            print(f"   {icon} Line {line_num}: {target}")
        print()
    
    if missing:
        print(f"❌ Found {len(missing)} BROKEN local references:")
        for ref_type, target, line_num in missing:
            icon = "🖼️" if ref_type == "image" else "🔗"
            print(f"   {icon} Line {line_num}: {target}")
        print()
        print("💡 Fix these paths or generate the missing files.")
        sys.exit(1)
    else:
        print(f"✅ All {len(found)} local references are valid!")
        sys.exit(0)


if __name__ == "__main__":
    main()
