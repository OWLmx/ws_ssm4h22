#!/bin/bash

REMOTE_DIR="~/projects/SMM4H22"
REMOTE_HOST="idsia_hpc"

function join { local IFS="$1"; shift; echo "$*"; }

Help()
{
   # Display Help
   echo "miniscript to sync project with remote cluster (e.g., HPC)."
   echo "** This script should be run from the project root (above src folder)"
   echo
   echo "Syntax: sybnc_remote [-g|h|v|V]"
   echo "options:"
   echo "h     Print this Help."
   echo "c     Sync code (src dir)"
   echo "d     Sync data dir"
   echo "p     Sync specific dir"
   echo "f     Files to filter in custom sync (p)"
   echo "y     by default sync is ran as DRY-RUN, include this opt to do the sync"
   echo
}

# Set variables
set -f  # disable (glob) expansion
DRY_RUN="n" # dry by default
TGT_DIR=""
TGTS=("") 
TGT_FILES='-. tracking/'


############################################################
# Process the input options. Add options as needed.        #
############################################################
# Get the options
while getopts ":hycdp:f:" option; do
   case $option in
      h) # display Help
         Help
         exit
        # return
         ;;
      y) # do the op
         DRY_RUN=""
         ;;
      c) # sync code
         TGT_DIR="src"
         TGTS+=( "merge scripts/sync_code.rules" )
         ;;         
      d) # sync code
        TGT_DIR="data"
         TGTS+=( "merge scripts/sync_data.rules" )
         ;;         
      p) # directory to be sync (relative to project root)
         echo "=> $OPTARG"
         TGT_DIR=$OPTARG
         ;;
      f) # files to be sync (*.* for all)
         TGT_FILES=$OPTARG
         ;;
     \?) # Invalid option
         echo "Error: Invalid option"
         exit
         ;;
   esac
done

if [ "$#" == 0 ]; then # default case
  TGT_DIR="src"
  TGTS+=( "merge scripts/sync_code.rules" )
fi

filters=$(join ' ' ${TGTS[@]})
echo rsync -arvzhP$DRY_RUN --filter="${filters}" -e ssh --progress ./$TGT_DIR/ $REMOTE_HOST:$REMOTE_DIR/$TGT_DIR
rsync -arvzhP$DRY_RUN --filter="${filters}" -e ssh --progress ./$TGT_DIR/ $REMOTE_HOST:$REMOTE_DIR/$TGT_DIR