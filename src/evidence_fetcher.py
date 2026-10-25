"""Real publisher thumbnails with URL/hash receipts; never generated evidence images."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .media import PipelineError


def verify_headline_excerpt(headline,publisher_headline):
    def normalized(text):
        return re.sub(r'\W+',' ',unicodedata.normalize('NFKC',text).casefold()).strip()
    if normalized(headline) not in normalized(publisher_headline):
        raise PipelineError('Evidence headline is not an excerpt of the real publisher headline')


class Metadata(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values={}

    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='meta':
            self.values[attrs.get('property',attrs.get('name',''))]=attrs.get('content','')


def public_url(url):
    import ipaddress
    import socket
    parsed=urlparse(url)
    if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password:
        raise PipelineError('Evidence requires a public HTTPS publisher URL')
    try:
        addresses=socket.getaddrinfo(parsed.hostname,443,type=socket.SOCK_STREAM)
        if any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
            raise PipelineError('Evidence URL resolves to a non-public address')
    except socket.gaierror as exc:
        raise PipelineError('Cannot resolve publisher evidence URL') from exc
    return url


class EvidenceFetcher:
    def __init__(self,root):
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=True)

    def fetch(self,url):
        key=hashlib.sha256(url.encode()).hexdigest()[:20]
        receipt=self.root/(key+'.json')
        if receipt.exists():
            info=json.loads(receipt.read_text(encoding='utf-8'))
            image=self.root/info['image_file']
            if image.exists() and hashlib.sha256(image.read_bytes()).hexdigest()==info['image_sha256']:
                return info|{'image_path':str(image)}
            raise PipelineError('Evidence cache changed; review provenance before using it')
        def download(address,limit):
            public_url(address)
            with urlopen(Request(address,headers={'User-Agent':'Mozilla/5.0'}),timeout=45) as response:
                public_url(response.url)
                data=response.read(limit+1)
            if len(data)>limit:
                raise PipelineError('Publisher evidence exceeds download limit')
            return data
        raw=download(url,8*1024*1024)
        parser=Metadata()
        parser.feed(raw.decode('utf-8',errors='replace'))
        photo=parser.values.get('og:image')
        title=parser.values.get('og:title','')
        if not photo or not title:
            raise PipelineError('Publisher has no real image/headline metadata; supply reviewed evidence')
        image=download(photo,20*1024*1024)
        import cv2
        import numpy as np
        decoded=cv2.imdecode(np.frombuffer(image,np.uint8),cv2.IMREAD_COLOR)
        if decoded is None or min(decoded.shape[:2])<120:
            raise PipelineError('Publisher thumbnail is missing, invalid or too small')
        target=self.root/(key+'.jpg')
        cv2.imwrite(str(target),decoded)
        info={'source_url':url,'publisher':parser.values.get('og:site_name') or urlparse(url).hostname,
              'publisher_headline':title,'image_url':photo,'image_file':target.name,
              'image_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
              'page_sha256':hashlib.sha256(raw).hexdigest(),
              'use':'Publisher thumbnail and attributed headline excerpt; does not verify every narration claim'}
        receipt.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        return info|{'image_path':str(target)}
