import os
import logging
from typing import List, Dict
from code_parser import CodeParser
from config import count_tokens, generate_unique_id, CHUNK_ENCODING


def chunk_code_structure_aware(code: str, token_limit: int, source_file: str, code_parser: CodeParser,
                               encoding_name: str = CHUNK_ENCODING) -> List[Dict]:
    logging.debug(f"Starting structure-aware chunking for {source_file} with limit {token_limit}")
    chunks = []

    # Early return for empty code
    if not code.strip():
        logging.debug("Skipping empty code.")
        return chunks

    # Split code into lines
    lines = code.splitlines()
    num_lines = len(lines)
    if num_lines == 0:
        return chunks

    # Parse code structure
    root_node = code_parser.parse_code(code)

    # Determine structural breakpoints
    if not root_node:
        logging.warning(f"Could not parse code for {source_file}. Using basic chunking strategy.")
        final_breakpoints = {0}  # Start with line 0 as a definite breakpoint
        comment_map = {}
    else:
        # Extract structure information
        breakpoints = code_parser.extract_lines_for_points_of_interest(root_node)
        comments = code_parser.extract_lines_for_comments(root_node)
        comment_set = set(comments)

        # Build comment-to-structure association map
        # This maps comments to the next structural element they're associated with
        comment_map = {}
        structure_indices = sorted(breakpoints)

        for comment_idx in sorted(comment_set):
            # Find the next structure point after this comment
            next_structures = [idx for idx in structure_indices if idx > comment_idx]
            if next_structures:
                closest_structure = min(next_structures)
                if closest_structure - comment_idx <= 3:  # Assume comments within 3 lines belong to next structure
                    comment_map[comment_idx] = closest_structure

        # Create intelligent breakpoints with comment awareness
        final_breakpoints = set([0])  # Always include start of file

        # Add all structural points as potential breakpoints
        for bp in breakpoints:
            # Include the breakpoint itself
            final_breakpoints.add(bp)

            # Find all comments associated with this structure point
            associated_comments = [c for c, s in comment_map.items() if s == bp]
            # Add the first associated comment as a breakpoint too
            if associated_comments:
                final_breakpoints.add(min(associated_comments))

        # Add potential breakpoints at logical code segments
        indent_levels = {}
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('#'):
                indent = len(line) - len(line.lstrip())
                indent_levels[i] = indent

        # Find places where indentation returns to 0 (potential new logical sections)
        for i in range(1, num_lines):
            if i in indent_levels and indent_levels[i] == 0 and i - 1 in indent_levels and indent_levels[i - 1] > 0:
                final_breakpoints.add(i)

    # Sort breakpoints for sequential processing
    final_breakpoints = sorted(list(final_breakpoints))
    logging.debug(f"Identified breakpoints at lines: {[b + 1 for b in final_breakpoints]}")

    # Precompute token counts for each line (with newline consideration)
    token_counts = []
    for i, line in enumerate(lines):
        # Add 1 token for newline character if not the last line
        newline_token = 1 if i < num_lines - 1 else 0
        token_counts.append(count_tokens(line, encoding_name) + newline_token)

    # Analyze line groups to predict optimal chunking points
    line_weights = [1] * num_lines  # Default weight

    # Increase weight for lines that are important to keep together
    for i in range(num_lines):
        # Increase weight for comment lines followed by code
        if i in comment_set and i + 1 < num_lines and i + 1 not in comment_set:
            line_weights[i] = 3

        # Increase weight for function/class definitions
        if i in breakpoints:
            line_weights[i] = 4

    # Chunking algorithm
    chunks = []
    start_line = 0
    chunk_number = 1

    i = 0
    while i < num_lines:
        # Initialize tracking for current chunk
        current_tokens = 0
        current_end = i

        # Keep adding lines until we hit a limit
        while current_end < num_lines:
            next_tokens = current_tokens + token_counts[current_end]

            # If adding this line would exceed the limit, stop
            if next_tokens > token_limit and current_tokens > 0:
                break

            current_tokens = next_tokens
            current_end += 1

        # Handle case where a single line exceeds token limit
        if current_end == i:
            logging.warning(f"Line {i + 1} exceeds token limit ({token_counts[i]} tokens). Creating oversized chunk.")
            current_end = i + 1

        # Find optimal split point if needed
        if current_end < num_lines:
            # Look for a good breakpoint within our current span
            valid_breaks = [bp for bp in final_breakpoints if i <= bp < current_end]

            if valid_breaks:
                # Use the latest valid breakpoint
                split_point = max(valid_breaks)

                # Avoid creating tiny chunks when close to full size
                if split_point == i and current_end - i > 3:
                    # Find the next best breakpoint
                    later_breaks = [bp for bp in valid_breaks if bp > i]
                    if later_breaks:
                        split_point = min(later_breaks)
            else:
                # No good breakpoint, just use the current end
                split_point = current_end - 1
        else:
            # We've reached the end of the file
            split_point = current_end - 1

        # Create chunk
        chunk_end = split_point + 1
        chunk_text = "\n".join(lines[start_line:chunk_end])

        if chunk_text.strip():  # Skip empty chunks
            chunk_id = generate_unique_id(f"chunk_{os.path.basename(source_file)}_{chunk_number}_")
            metadata = {
                "source_file": source_file,
                "language": "python",
                "chunk_index": chunk_number,
                "start_line": start_line + 1,
                "end_line": chunk_end,
                "token_count": sum(token_counts[start_line:chunk_end])
            }
            chunks.append({"id": chunk_id, "text": chunk_text, "metadata": metadata})
            logging.debug(f"Created chunk {chunk_number}: Lines {metadata['start_line']}-{metadata['end_line']}, "
                          f"Tokens: {metadata['token_count']}")
            chunk_number += 1

        # Move to next starting point
        start_line = chunk_end
        i = chunk_end

    # Additional logging info
    avg_tokens = sum(len(c["text"].split()) for c in chunks) / len(chunks) if chunks else 0
    logging.info(f"Finished chunking {source_file}: {len(chunks)} chunks, avg ~{avg_tokens:.1f} tokens/chunk")

    return chunks