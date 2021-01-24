export HEROKU_APIKEY_=`echo ${HEROKU_APIKEY}|sed 's/_/-/g'`

echo machine api.heroku.com > ~/.netrc
echo "  login ${GITLAB_USER_EMAIL}" >> ~/.netrc
echo "  password ${HEROKU_APIKEY_}" >> ~/.netrc

echo machine git.heroku.com >> ~/.netrc
echo "  login ${GITLAB_USER_EMAIL}" >> ~/.netrc
echo "  password ${HEROKU_APIKEY_}" >> ~/.netrc
git config --global user.email ${GITLAB_USER_EMAIL}
git config --global user.name ${GITLAB_USER_EMAIL}

heroku git:clone -a vtdocs
ls -la
unzip build/docs/VinetalkDocs.zip &> /dev/null
cd VinetalkDocs
echo "<?php header('Location: index.html'); ?>" > index.php
sed -i '/carvgit/d' index.html
cd -
cp -r VinetalkDocs/* vtdocs/
cd vtdocs
echo '{}' > composer.json
git add .
git commit -m "Updated docs"
git push heroku master
