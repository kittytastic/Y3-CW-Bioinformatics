import argparse
import os

def assertFileExists(path):
    if not os.path.isfile(path):
        print("ERROR: %s doesn't exist."%path) 
        exit()

if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='Encoder')
    parser.add_argument("file_path")
    parser.add_argument("-t", dest="trial", help="Trial run - won't store output", action='store_true')
    parser.add_argument("--type", dest="testType", help="Select a test type from: %s"%(",".join(testSuites.keys())),default="live" )
    
    args = parser.parse_args()
    file_path = str(args.file_path)
    os.environ['TRIAL'] = "T" if args.trial else "F"
    
    # Do something
