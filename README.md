# CS252S22ProjectsLabs
Template for CS252 Spring 22 Projects and Labs

Some Git Instructions
---------------------

* Fork this git repository (get your own copy): log in to [github](https://github.com) and go to [this page](https://github.com/ajstent/CS252ProjectsLabs) and click "Fork"
  * In Settings:
    * Make sure you set the repository to 'Private'
    * Make sure you add your instructor (ajstent) as a Collaborator
* Set a ssh key for github: follow [these instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
* Work with your repository:
  * Using git from github.com:
    * Open your fork (copy) of the repository in a GitHub code space
    * Don't forget to commit your work before the code space shuts down!
  * Using git on the commandline (including in a jupyterhub terminal):
    * Get your fork (copy) of the repository: git clone git@github.com:(yourusername)/CS252ProjectsLabs.git
    * Update your repository after you have...
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
  * Using git from jupyterhub (you may use cs251.jupyter.colby.edu):
    * first, you might need to set a ssh token in github
    * then, you need to clone (checkout) the repository
      * in the terminal window, make sure you are in the folder where you want to be and type: 
        * git clone git@github.com:(yourusername)/CS252ProjectsLabs.git 
      * don't forget to commit your code before you exit the browser!
