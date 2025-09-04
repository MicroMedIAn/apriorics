#!/usr/bin/env bash

set -euo pipefail

FILE_EXT=".svs"

# Globals
INPUT_DIR=""
OUTPUT_DIR=""
NAME_LIST_FILE=""
SLIDE_NUMBER=""
declare -a FILENAMES

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
            -n|--number)
                SLIDE_NUMBER="$2"
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

    if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" || -z "$NAME_LIST_FILE" || -z "$SLIDE_NUMBER" ]]; then
        printf "Usage: %s -i <input_folder> -o <user@ip:/output_folder> -l <filename_list> -n <number>\n" "$0" >&2
        return 1
    fi
}

sanitize_paths() {
    INPUT_DIR=$(realpath "$INPUT_DIR")
    NAME_LIST_FILE=$(realpath "$NAME_LIST_FILE")

    if [[ ! -d "$INPUT_DIR" ]]; then
        printf "Error: Input folder does not exist: %s\n" "$INPUT_DIR" >&2
        return 1
    fi

    if [[ ! -f "$NAME_LIST_FILE" ]]; then
        printf "Error: Filename list does not exist: %s\n" "$NAME_LIST_FILE" >&2
        return 1
    fi

    if [[ ! "$SLIDE_NUMBER" =~ ^[0-9]+$ ]]; then
        printf "Error: Slide number must be numeric: %s\n" "$SLIDE_NUMBER" >&2
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

is_in_list() {
    local target="$1"
    local name
    for name in "${FILENAMES[@]}"; do
        if [[ "$target" == "$name" ]]; then
            return 0
        fi
    done
    return 1
}

remote_file_exists() {
    local remote_path="$1"
    local remote_host="${OUTPUT_DIR%%:*}"
    local remote_dir="${OUTPUT_DIR#*:}"

    if ssh -o BatchMode=yes -o ConnectTimeout=10 "$remote_host" "[ -f \"$remote_dir/$remote_path\" ]"; then
        return 0
    fi
    return 1
}

process_files() {
    local file
    shopt -s nullglob
    for file in "$INPUT_DIR"/*-"$SLIDE_NUMBER"_*"$FILE_EXT"; do
        local basename stem slidename
        basename=$(basename "$file")
        stem="${basename%$FILE_EXT}"
        slidename="${stem%%-*}"

        if ! is_in_list "$slidename"; then
            continue
        fi

        local dest="$slidename$FILE_EXT"
        if remote_file_exists "$dest"; then
            continue
        fi

        printf "%s\n" "$slidename"

        if ! rsync -a "$file" "$OUTPUT_DIR/$dest"; then
            printf "Error: rsync failed for %s\n" "$file" >&2
            return 1
        fi
    done
    shopt -u nullglob
}

main() {
    parse_arguments "$@" || return 1
    sanitize_paths || return 1
    read_filename_list || return 1
    process_files || return 1
}

main "$@"
