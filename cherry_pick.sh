#!/bin/bash

# Définir la branche de départ et les branches cibles directement dans le script
TARGET_BRANCHES=("master" "ae1ae3" "phh3" "ERGPodoplanine" "P40ColIV" "cd3cd20")  # Liste des branches cibles

IGNORE_BRANCHES=("$@")
# Obtenir la branche actuelle
SOURCE_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Obtenir le dernier commit de la branche de départ
LAST_COMMIT=$(git rev-parse HEAD)

echo "Branche de départ : $SOURCE_BRANCH"
echo "Dernier commit : $LAST_COMMIT"

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
    
    # Tenter le cherry-pick du dernier commit
    git cherry-pick "$LAST_COMMIT"
    
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