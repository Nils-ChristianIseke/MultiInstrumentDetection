# Joining the SDS
- you need to be invited to join SDS with **acronym and password**. Please provide your uni-id to your supervisor
- access SDS management via this link: https://sds-hd.urz.uni-heidelberg.de/management/
    - unter Mitarbeit Speichervorhaben bei einem existierenden SV beitreten.
    - chose Uni-Heidelberg in dropdown menu and login with normal credentials
      - sign up using SV-credentials and personal data; accept terms of use
    - Check E-mail for Registration.
        - first E-mail (should arrive immediatly) indicates that membership of SDS was succesfully processed![[Pasted image 20220211091845.png]]
        - second E-mail (arrived 25 min later) confirmed registration![[Pasted image 20220211092136.png]]


## (optional) Mount SDS-storage to computer
- follow this guide: https://wiki.bwhpc.de/e/Sds-hd_CIFS#Using_SMB.2FCIFS_for_Windows_client
### WINDOWS:
	- add file-path directly into the file-path of windows explorer ![[Pasted image 20220211092751.png]]
	- might take some time for login-windows to appear

### LINUX
  - sd22a004 is the name of the SV
  - create the folder/s you want to mount at, with root privileges and
  - chown it afterwards with your 'normal' user --> replace %USERID% with your local userid  
>  sudo mkdir /mnt/sds-stud  
  sudo chown -R %USERID% /mnt/sds-stud

Map the sds to this created folder  
> sshfs -o reconnect hd_qz262@lsdf02-sshfs.urz.uni-heidelberg.de:/sd22a004 /mnt/sds-stud -o idmap=user -o uid=\$(id -u) -o gid=$(id -g)


# bwservices
- make sure to be in the uni-network. You might want to use a VPN for this if you are from home. If not don't have one yet: https://www.urz.uni-heidelberg.de/de/support/anleitungen/anyconnect-fuer-vpn-installieren
- visit https://bwservices.uni-heidelberg.de/ and register for SDS (and bwVisu if needed)
- set Passwords for registered projects
- hover over "Übersicht" and click on "Meine Tokens" to set up two-factor authentication with prefered method
- Your login name for all bwservices is hd_uni-ID , for example hd_xy123


## set up bwVisu
- Log in to the bwVisu web frontend at https://bwvisu-web.urz.uni-heidelberg.de. Your username will be <site-prefix>_<uni-id>, e.g. hd_ab123 for a user from Heidelberg.
- The password will be your bwVisu service password set at bwservices.uni-heidelberg.de, and your registered device will be used as second factor.
- scroll down application window and click on "latest" for jupyter ![image info](./attachments/Pasted image 20220211093328.png)
- Click on green button "Start new job"
- Setup Job settings (you might want to increase the runtime, but make sure to properly shut down the job afterwards to free up ressources)
- click on your running job (blue button)
- click on your jupyter-connection-link.
- verify that you are able to see the sds-hd drive mounted

## Installing env and packages
- Note: it's normal if the import of libraries takes long in bwVisu, so please be patient
- load miniconda modules and the  cuda modules available by clicking on the hexagon icon on the left panel
- Open a terminal from Jupyter. This can be done by clicking the + button from the top left and selecting Terminal, and then type the following commands:


> conda create --name wft_endo  
conda activate wft_endo  
pip install -r /mnt/sds-hd/sd22a004/guest/setup/requirements.txt  
pip install ipykernel  
python3 -m ipykernel install --user --name=wft_endo  

- hint: if 'conda activate wft_endo' doesn't work, try 'source activate wft_endo' !
- try to run the cells in endo_data_quickstart and pytorch_quickstart notebooks
- once successful, please go back to the job tab and cancel the job to free up the resource for other users
