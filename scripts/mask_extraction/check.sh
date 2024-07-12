SCRIPT_NAME="run.sh"

# Get the count of processes with the script name excluding the grep command itself
PROCESS_COUNT=$(ps aux | grep "$SCRIPT_NAME" | grep -v "grep" | wc -l)

# Check if the process count is greater than 0
if [ "$PROCESS_COUNT" -eq 0 ]; then
  source /home/Robin/.bashrc
  docker kill $(docker ps -q)
  eval "$(conda shell.bash hook)"
  conda activate apriorics
  cd /home/Robin/apriorics/scripts/mask_extraction/
  bash run.sh
fi
