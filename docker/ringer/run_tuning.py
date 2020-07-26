#!/usr/bin/env python


from Gaugi.messenger import LoggingLevel, Logger
import argparse
import sys,os


mainLogger = Logger.getModuleLogger("job")
parser = argparse.ArgumentParser(description = '', add_help = False)
parser = argparse.ArgumentParser()


parser.add_argument('-c','--configFile', action='store',
        dest='configFile', required = True,
            help = "The job config file that will be used to configure the job (sort and init).")

parser.add_argument('-d','--dataFile', action='store',
        dest='dataFile', required = False, default = None,
            help = "The data/target file used to train the model.")

parser.add_argument('-r','--refFile', action='store',
        dest='refFile', required = False, default = None,
            help = "The reference file.")

parser.add_argument('-t', '--tag', action='store',
        dest='tag', required = True, default = None,
            help = "The tuning tag in the tuning branch in ringertunings repository")

parser.add_argument('-b', '--branch', action='store',
        dest='branch', required = True, default = None,
            help = "The tuning branch in ringetunings repository")

parser.add_argument('-v', '--volume', action='store',
        dest='volume', required = True, default = None,
            help = "The output path")



if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

args = parser.parse_args()

volume = args.volume


commands = open(volume+'/commands.sh', 'w')
# Update all pip packages
# commands.write('source /setup_envs.sh\n')
# Download the ringer tuning repository
commands.write("cd %s && git clone https://github.com/jodafons/ringer.git && cd ringer && git checkout %s && cd versions/%s\n"%(voume, args.branch,args.tag))
commands.write("python job_tuning.py -d %s -o %s -c %s -r %s\n"%(args.dataFile, volume, args.configFile, args.refFile) )
commands.close()

os.system('source %s/commands.sh'%volume)
os.system( 'rm -rf %s/ringer'%volume)
os.system( 'rm -rf %s/command.sh'%volume)










