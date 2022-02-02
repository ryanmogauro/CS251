# CS252S22ProjectsLabs
Template for CS252 Spring 22 Projects and Labs

Some Git Instructions
---------------------

* Forking this git repository (get your own copy): log in to [github](https://github.com) and go to [this page](https://github.com/ajstent/CS252S22ProjectsLabs) and click "Fork"
  * In Settings:
    * Make sure you set the repository to 'Private'
    * Make sure you add (ajstent) as a Collaborator
* Using git on the commandline (including in a jupyterhub terminal):
  * Getting your fork (copy) of the repository: git checkout git@github.com:<yourusername>/CS252S22ProjectsLabs.git
  * Updating your repository after you have...
    * Changed a file: 
      * git commit -m 'This is how I changed these files' .
      * git push origin main
    * Added a file
      * git add <file I added>
      * Then see "Changed a file"
    * Removed a file
      * git rm <file I want to go away>
      * Then see "Changed a file"
* Using git from VSCode: [docs](https://docs.microsoft.com/en-us/learn/modules/use-git-from-vs-code/)
* Using git from jupyterhub:
