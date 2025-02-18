# --- cmd tools --- #

# git add & commit & push
git_all() {
	git add .
	git commit -m "$1"
	git push
}

gpul() {
	git pull
}

ifs(){
	# cd to mcts workspace and then innitialize file structure
	echo 'initialize directories for '$1
	mkdir $1/.temp
	mkdir $1/.log
}

gset() {
	if [ -f '/inspire' ]; then
		# only reset on inspire cluster
		git reset --hard HEAD
	fi
}
