#!/bin/sh
#for i in `seq 1 10`;
#do
#./halite --replay-directory replays/ -vvv --width 32 --height 32 "python3 ../old_bot/MyBot.py" "python3 MyBot.py"
#done

for i in `seq 1 10`;
do
./halite --replay-directory replays/ -vvv --width 64 --height 64 "python3 ../old_bot/MyBot.py" "python3 MyBot.py"
done
