#!/bin/bash

# Check if at least 3 arguments are provided (source_branch, source_commit, and at least one target_branch)
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 source_branch source_commit target_branch1 target_branch2 ... target_branchn"
    exit 1
fi

# Assign the first argument as the source branch and the second as the commit to cherry-pick
SOURCE_BRANCH=$1
SOURCE_COMMIT=$2

# Shift the arguments to get the list of target branches
shift 2
TARGET_BRANCHES=("$@")

# Fetch the latest branches information from remote
git fetch origin

# Checkout to the source branch and make sure it's up-to-date
git checkout "$SOURCE_BRANCH"
git pull origin "$SOURCE_BRANCH"

# Loop through each target branch and cherry-pick the commit
for TARGET_BRANCH in "${TARGET_BRANCHES[@]}"; do
    echo "Processing branch: $TARGET_BRANCH"
    
    # Checkout the target branch and make sure it's up-to-date
    git checkout "$TARGET_BRANCH"
    git pull origin "$TARGET_BRANCH"
    
    # Attempt to cherry-pick the commit
    git cherry-pick "$SOURCE_COMMIT"
    
    # Check if there was a conflict
    if [ $? -ne 0 ]; then
        echo "Merge conflict detected in branch: $TARGET_BRANCH"
        echo "Please resolve the conflict manually."
        exit 1
    fi

    # Push the changes to the target branch
    git push origin "$TARGET_BRANCH"
done

# Checkout back to the source branch
git checkout "$SOURCE_BRANCH"

echo "Cherry-pick completed successfully for all branches."
