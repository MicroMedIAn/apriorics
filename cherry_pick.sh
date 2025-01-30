#!/bin/bash

# Définir la branche de départ et les branches cibles directement dans le script
TARGET_BRANCHES=("master" "ae1ae3" "phh3" "ERGPodoplanine" "P40ColIV" "cd3cd20")  # Liste des branches cibles

# Function to display usage
usage() {
    echo "Usage: $0 [--commit <commit-hash>] [branches-to-ignore...]"
    exit 1
}

# Parse command-line arguments
COMMIT_HASH=""
IGNORE_BRANCHES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --commit)
            COMMIT_HASH="$2"
            shift 2
            ;;
        *)
            IGNORE_BRANCHES+=("$1")
            shift
            ;;
    esac
done

# Obtenir la branche actuelle
SOURCE_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Définir le commit à utiliser (soit celui fourni, soit le dernier commit de la branche actuelle)
if [ -z "$COMMIT_HASH" ]; then
    COMMIT_HASH=$(git rev-parse HEAD)
fi

echo "Branche de départ : $SOURCE_BRANCH"
echo "Commit à appliquer : $COMMIT_HASH"

# Fetch les dernières informations des branches distantes
git fetch origin

# Loop pour chaque branche cible
for TARGET_BRANCH in "${TARGET_BRANCHES[@]}"; do
    # Ignorer la branche source si elle est dans la liste des branches cibles
    if [ "$TARGET_BRANCH" == "$SOURCE_BRANCH" ]; then
        echo "Ignorer la branche de départ '$SOURCE_BRANCH'."
        continue
    fi

    if [[ " ${IGNORE_BRANCHES[@]} " =~ " ${TARGET_BRANCH} " ]]; then
        echo "Branche '$TARGET_BRANCH' ignorée."
        continue
    fi

    echo "Traitement de la branche : $TARGET_BRANCH"
    
    # Basculer sur la branche cible et s'assurer qu'elle est à jour
    git checkout "$TARGET_BRANCH"
    git pull origin "$TARGET_BRANCH"
    
    # Tenter le cherry-pick du commit spécifié
    git cherry-pick "$COMMIT_HASH"
    
    # Vérifier s'il y a un conflit
    if [ $? -ne 0 ]; then
        echo "Conflit de fusion détecté sur la branche : $TARGET_BRANCH"
        echo "Veuillez résoudre le conflit manuellement."
        exit 1
    fi

    # Pousser les modifications sur la branche cible
    git push origin "$TARGET_BRANCH"
done

# Revenir à la branche de départ
git checkout "$SOURCE_BRANCH"

echo "Cherry-pick terminé avec succès pour toutes les branches."