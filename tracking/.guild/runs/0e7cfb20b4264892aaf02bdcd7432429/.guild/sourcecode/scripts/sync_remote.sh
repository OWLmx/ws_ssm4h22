#!/bin/bash

REMOTE_DIR="~/projects/SMM4H22"
REMOTE_HOST="idsia_hpc"

# n means dry-run (nothing is actually done). empty to do the work
# DRY_RUN="$1"
# TGT_DIR=${1:-"src"}    
# DRY_RUN=${2:-"n"}

# push to remote
# -W transmit whole files and not compute differences in bytes (best when a fast network)
# --delete delete on target if deleted (non existing) in source
# -u update, skip files that are newer in the receiver

# To sync
# - pull with -update without delete
# - push with -update with -delete (remove removed runs)
#rsync -arvzhP$DRY_RUN -e ssh --progress $REMOTE_HOST:$REMOTE_DIR/ .
#rsync -arvzhP$DRY_RUN --delete -e ssh --progress ./ $REMOTE_HOST:$REMOTE_DIR

# it should be -> 
# pull runs newer than the last sync
# push all & delete - -> 
# ?? pull all (sync code as well)


# seems to be more convinient to push and pull runs using guild
# if [[ -f "./tracking/.last_sync" ]]; then # only if not the first push
#     # pull new remote runs
#     SOURCE="$REMOTE_DIR/tracking"
#     rsync -arvzhP$DRY_RUN \
#         --update \
#         --files-from=<(ssh $REMOTE_HOST "find $SOURCE -type f -newer $SOURCE/.last_sync -exec realpath --relative-to=$SOURCE '{}' \;") \
#         -e ssh --progress $REMOTE_HOST:$REMOTE_DIR/ .

#     # another sync to pull code??
# fi

# touch "./tracking/.last_sync" # update last_sync

# push (runs & all), -> delete in remote runs not in local (previously downloaded)
# rsync -arvzhP$DRY_RUN --delete -e ssh --progress ./ $REMOTE_HOST:$REMOTE_DIR
# rsync -arvzhP$DRY_RUN -e ssh --progress ./ $REMOTE_HOST:$REMOTE_DIR

# sync only code (no guild config)
# rsync -arvzhP$DRY_RUN -e ssh --progress ./$1 $REMOTE_HOST:$REMOTE_DIR/$1

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
# # alternative to set -f
# set -o noglob
# # undo it by 
# set +o noglob
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


# if [ -z "$TGT_FILES" ]
# then
#       SRC="${TGT_DIR}"
# else
#       SRC="${TGT_DIR}/${TGT_FILES}"
# fi

# echo rsync -arvzhP$DRY_RUN -e ssh --progress ./$SRC $REMOTE_HOST:$REMOTE_DIR/$TGT_DIR
# echo rsync -arvzhP$DRY_RUN --filter="${TGT_FILES}" -e ssh --progress ./$TGT_DIR $REMOTE_HOST:$REMOTE_DIR/$TGT_DIR
# rsync -arvzhP$DRY_RUN --filter="${TGT_FILES}" -e ssh --progress ./$TGT_DIR $REMOTE_HOST:$REMOTE_DIR/$TGT_DIR

# echo rsync -arvzhP$DRY_RUN --filter="merge scripts/sync_code.rules" -e ssh --progress ./$TGT_DIR $REMOTE_HOST:$REMOTE_DIR/$TGT_DIR
# rsync -arvzhP$DRY_RUN --filter="merge scripts/sync_code.rules" -e ssh --progress ./$TGT_DIR $REMOTE_HOST:$REMOTE_DIR/$TGT_DIR

# for t in ${TGTS[@]}; do
#   echo $t
# done

filters=$(join ' ' ${TGTS[@]})
echo rsync -arvzhP$DRY_RUN --filter="${filters}" -e ssh --progress ./$TGT_DIR/ $REMOTE_HOST:$REMOTE_DIR/$TGT_DIR
rsync -arvzhP$DRY_RUN --filter="${filters}" -e ssh --progress ./$TGT_DIR/ $REMOTE_HOST:$REMOTE_DIR/$TGT_DIR