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
unzip build/docs/AraxDocs.zip &> /dev/null
cd AraxDocs
echo "<?php header('Location: index.html'); ?>" > index.php
sed -i '/carvgit/d' index.html
cd -
cd vtdocs
git rm -rq *
cd -
cp -r AraxDocs/* vtdocs/
cd vtdocs
echo '{}' > composer.json
git add --all .
if [ "${CI_COMMIT_REF_NAME}" == "master" ]
then
	git commit -m "Updated docs"
	git push heroku master
fi
