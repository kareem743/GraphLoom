import os
import logging
from typing import List, Dict
from code_parser import CodeParser
from config import count_tokens, generate_unique_id, CHUNK_ENCODING

def chunk_code_structure_aware(code: str, token_limit: int, source_file: str, code_parser: CodeParser, encoding_name: str = CHUNK_ENCODING) -> List[Dict]:
    logging.debug(f"Starting structure-aware chunking for {source_file} with limit {token_limit}")
    chunks = []
    if not code.strip():
        logging.debug("Skipping empty code.")
        return chunks

    root_node = code_parser.parse_code(code)
    if not root_node:
        logging.warning(f"Could not parse code for {source_file}. Chunking may be suboptimal (using line breaks only).")
        final_breakpoints = [0]
        comment_set = set()
    else:
        lines = code.splitlines()
        num_lines = len(lines)
        if num_lines == 0: return chunks

        breakpoints_indices = code_parser.extract_lines_for_points_of_interest(root_node)
        comment_indices = code_parser.extract_lines_for_comments(root_node)
        comment_set = set(comment_indices)

        logging.debug(f"Initial breakpoints (0-based indices): {breakpoints_indices}")
        logging.debug(f"Comment/decorator lines (0-based indices): {comment_indices}")

        adjusted_breakpoints = set()
        for bp_idx in breakpoints_indices:
            current_line_idx = bp_idx - 1
            actual_bp_idx = bp_idx
            while current_line_idx >= 0 and current_line_idx in comment_set:
                actual_bp_idx = current_line_idx
                current_line_idx -= 1
            adjusted_breakpoints.add(actual_bp_idx)

        adjusted_breakpoints.add(0)
        final_breakpoints = sorted(list(adjusted_breakpoints))
        logging.debug(f"Final adjusted breakpoints (0-based indices): {final_breakpoints}")

    lines = code.splitlines()
    num_lines = len(lines)
    if num_lines == 0: return chunks

    current_chunk_lines = []
    current_token_count = 0
    start_line_idx = 0
    chunk_number = 1

    i = 0
    while i < num_lines:
        line = lines[i]
        line_token_count = count_tokens(line, encoding_name) + (1 if i < num_lines - 1 else 0)

        if current_token_count > 0 and current_token_count + line_token_count > token_limit:
            logging.debug(f"Line {i+1} ('{line[:30]}...') exceeds token limit ({current_token_count} + {line_token_count} > {token_limit})")
            possible_stops = [bp for bp in final_breakpoints if start_line_idx <= bp < i]

            if possible_stops:
                stop_line_idx = max(possible_stops)
                logging.debug(f"Found suitable breakpoint at index {stop_line_idx} (line {stop_line_idx+1})")
            else:
                if not current_chunk_lines:
                    logging.warning(f"Line {i+1} itself exceeds token limit ({line_token_count}). Creating oversized chunk.")
                    stop_line_idx = i + 1
                else:
                    logging.debug(f"No breakpoint found between {start_line_idx} and {i}. Chunking before line {i+1}.")
                    stop_line_idx = i

            chunk_text = "\n".join(lines[start_line_idx:stop_line_idx])
            if chunk_text.strip():
                chunk_id = generate_unique_id(f"chunk_{os.path.basename(source_file)}_{chunk_number}_")
                metadata = {
                    "source_file": source_file, "language": "python", "chunk_index": chunk_number,
                    "start_line": start_line_idx + 1, "end_line": stop_line_idx
                }
                chunks.append({"id": chunk_id, "text": chunk_text, "metadata": metadata})
                logging.debug(f"Created chunk {chunk_number}: Lines {metadata['start_line']}-{metadata['end_line']}")
                chunk_number += 1

            start_line_idx = stop_line_idx
            current_chunk_lines = []
            current_token_count = 0
            i = max(i + 1, stop_line_idx)
            continue

        current_chunk_lines.append(line)
        current_token_count += line_token_count
        if (i + 1) in final_breakpoints and current_chunk_lines:
            logging.debug(f"Next line ({i+2}) is a breakpoint. Finalizing current chunk.")
            chunk_text = "\n".join(current_chunk_lines)
            chunk_id = generate_unique_id(f"chunk_{os.path.basename(source_file)}_{chunk_number}_")
            metadata = {
                "source_file": source_file, "language": "python", "chunk_index": chunk_number,
                "start_line": start_line_idx + 1, "end_line": i + 1
            }
            chunks.append({"id": chunk_id, "text": chunk_text, "metadata": metadata})
            logging.debug(f"Created chunk {chunk_number}: Lines {metadata['start_line']}-{metadata['end_line']} (breakpoint split)")
            chunk_number += 1

            start_line_idx = i + 1
            current_chunk_lines = []
            current_token_count = 0

        i += 1

    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines)
        if chunk_text.strip():
            chunk_id = generate_unique_id(f"chunk_{os.path.basename(source_file)}_{chunk_number}_")
            metadata = {
                "source_file": source_file, "language": "python", "chunk_index": chunk_number,
                "start_line": start_line_idx + 1, "end_line": num_lines
            }
            chunks.append({"id": chunk_id, "text": chunk_text, "metadata": metadata})
            logging.debug(f"Created final chunk {chunk_number}: Lines {metadata['start_line']}-{metadata['end_line']}")

    logging.info(f"Finished structure-aware chunking for {source_file}. Created {len(chunks)} chunks.")
    return chunks