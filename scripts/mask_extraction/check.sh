SCRIPT_NAME="run.sh"

# Get the count of processes with the script name excluding the grep command itself
PROCESS_COUNT=$(ps aux | grep "$SCRIPT_NAME" | grep -v "grep" | wc -l)

# Check if the process count is greater than 0
if [ "$PROCESS_COUNT" -eq 0 ]; then
  source /home/Robin/.bashrc
  __conda_setup="$('/opt/anaconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
  if [ $? -eq 0 ]; then
      eval "$__conda_setup"
  else
      if [ -f "/opt/anaconda/etc/profile.d/conda.sh" ]; then
          . "/opt/anaconda/etc/profile.d/conda.sh"
      else
          export PATH="/opt/anaconda/bin:$PATH"
      fi
  fi
  unset __conda_setup
  docker kill $(docker ps -q)
  eval "$(conda shell.bash hook)"
  conda activate apriorics
  cd /home/Robin/apriorics/scripts/mask_extraction/
  bash run.sh
fi
