#!/usr/bin/env bash

set -euo pipefail

FILE_EXTENSIONS=("*.gpkg") # Extend as needed

# Globals
INPUT_DIR=""
OUTPUT_DIR=""
NAME_LIST_FILE=""

parse_arguments() {
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            -i|--input)
                INPUT_DIR="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -l|--list)
                NAME_LIST_FILE="$2"
                shift 2
                ;;
            -*)
                printf "Unknown option: %s\n" "$1" >&2
                return 1
                ;;
            *)
                shift
                ;;
        esac
    done

    if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" || -z "$NAME_LIST_FILE" ]]; then
        printf "Usage: %s -i <input_folder> -o <output_folder> -l <filename_list>\n" "$0" >&2
        return 1
    fi
}

sanitize_paths() {
    INPUT_DIR=$(realpath "$INPUT_DIR")
    OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
    NAME_LIST_FILE=$(realpath "$NAME_LIST_FILE")

    if [[ ! -d "$INPUT_DIR" ]]; then
        printf "Error: Input folder does not exist: %s\n" "$INPUT_DIR" >&2
        return 1
    fi

    mkdir -p "$OUTPUT_DIR"

    if [[ ! -f "$NAME_LIST_FILE" ]]; then
        printf "Error: Filename list does not exist: %s\n" "$NAME_LIST_FILE" >&2
        return 1
    fi
}

read_filename_list() {
    local filenames_raw; filenames_raw=$(<"$NAME_LIST_FILE")
    if [[ -z "${filenames_raw// }" ]]; then
        printf "Error: Filename list is empty\n" >&2
        return 1
    fi
    mapfile -t FILENAMES < <(printf '%s\n' "$filenames_raw" | sed -E 's/^\s+|\s+$//g' | grep -Ev '^\s*$' | sort -u)
    if [[ "${#FILENAMES[@]}" -eq 0 ]]; then
        printf "Error: No valid filenames found in list\n" >&2
        return 1
    fi
}

build_rsync_include_file() {
    local include_file="$1"
    : > "$include_file"

    local name;
    for name in "${FILENAMES[@]}"; do
        if [[ -z "$name" ]]; then
            continue
        fi
        for ext in "${FILE_EXTENSIONS[@]}"; do
            printf -- "+ /%s\n" "${name}${ext#\*}" >> "$include_file"
        done
    done

    printf -- "- *\n" >> "$include_file"
}

sync_files() {
    local include_file; include_file=$(mktemp)
    if ! build_rsync_include_file "$include_file"; then
        rm -f "$include_file"
        printf "Error: Failed to build rsync include file\n" >&2
        return 1
    fi

    if ! rsync -av --include-from="$include_file" --exclude='*' "$INPUT_DIR"/ "$OUTPUT_DIR"/; then
        rm -f "$include_file"
        printf "Error: rsync operation failed\n" >&2
        return 1
    fi

    rm -f "$include_file"
}

main() {
    parse_arguments "$@" || return 1
    sanitize_paths || return 1
    read_filename_list || return 1
    sync_files || return 1
}

main "$@"
