NEW_UUID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)
echo $NEW_UUID
git add -A
git commit -m "$NEW_UUID"
git pull
git push
