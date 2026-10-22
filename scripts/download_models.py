#!/usr/bin/env python3
"""Download the released model containers, verify SHA256, and extract safely."""
import argparse,hashlib,shutil,tarfile,tempfile,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPOSITORY='https://github.com/shannonchen37/HyperMIL'

def download(url,path):
    request=urllib.request.Request(url,headers={'User-Agent':'HyperMIL-model-downloader'})
    with urllib.request.urlopen(request,timeout=120) as response,path.open('wb') as output:
        shutil.copyfileobj(response,output,8*1024*1024)

def install_archive(archive,root):
    root=root.resolve()
    with tarfile.open(archive,'r') as tar:
        members=tar.getmembers()
        for member in members:
            target=(root/member.name).resolve()
            if not member.isfile() or not target.is_relative_to(root/'checkpoints'):
                raise ValueError(f'Unexpected archive entry: {member.name}')
            if target.suffix not in ('.pt','.csv'):
                raise ValueError(f'Unexpected model asset: {member.name}')
        for member in members:
            target=root/member.name;target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as source,target.open('wb') as output:
                shutil.copyfileobj(source,output)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cohort',default='all',choices=['all','stad','thca','chol','lihc'])
    parser.add_argument('--release',default='v1.0.0')
    parser.add_argument('--output-root',type=Path,default=ROOT)
    args=parser.parse_args()
    cohorts=['stad','thca','chol','lihc'] if args.cohort=='all' else [args.cohort]
    for cohort in cohorts:
        name=f'hypermil_weights_{cohort}.tar'
        url=f'{REPOSITORY}/releases/download/{args.release}/{name}'
        with tempfile.TemporaryDirectory(prefix='hypermil-download-') as folder:
            archive=Path(folder)/name;checksum=Path(folder)/(name+'.sha256')
            print(f'Downloading {name}',flush=True)
            download(url+'.sha256',checksum);download(url,archive)
            expected=checksum.read_text().split()[0];digest=hashlib.sha256()
            with archive.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
            if digest.hexdigest()!=expected:raise ValueError(f'SHA256 mismatch: {name}')
            install_archive(archive,args.output_root)
        print(f'Installed and verified {cohort.upper()} models',flush=True)

if __name__=='__main__':main()
