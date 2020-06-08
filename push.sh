#!/bin/bash 

echo Enter commit message

read msg 

git add . 

git commit -m $msg$

git push origin master

hugo -d ../araza6.github.io

cd ../araza6.github.io

git add . 

git commit -m $msg$

git push origin master


