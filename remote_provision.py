import subprocess
import argparse
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--accesstoken",
    default=os.path.expanduser("~/.githubaccesstoken"),
    help="text file containing github acccess token",
)
parser.add_argument(
    "--apitoken",
    default=os.path.expanduser("~/.ghtoken"),
    help="text file containing github API token",
)

parser.add_argument("remote", help="remote host login, e.g. ubuntu@192.168.0.0")
parser.add_argument("myname", help='for git config, e.g. "John Doe"')
parser.add_argument("myemail", help="for git config, e.g. johndoe@email.com")

args = parser.parse_args()

myname = f'\\"{args.myname}\\"'
myemail = f'\\"{args.myemail}\\"'

with open(args.accesstoken) as f:
    token = f'\\"{f.readline().rstrip()}\\"'

scpcmd = f"scp provisioning/provision.sh {args.remote}:~; scp {args.apitoken} {args.remote}:~"
sshcmd = f'ssh {args.remote} "chmod 755 provision.sh; GHTOKEN={token} MYNAME={myname} MYEMAIL={myemail} sh provision.sh"'
cmd = f"{scpcmd} && {sshcmd}"
print("\t" + cmd)
res = input("Execute? (Y|n) ")
if res not in ["n", "N"]:
    subprocess.call(cmd, shell=True)
