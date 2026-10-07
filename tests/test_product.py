import json
import tempfile
import zipfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import pandas as pd
from PIL import Image
from src.product import write_json, catalog, edit_project, export_project, split_photo_conflicts
import numpy as np
from src.downloads import prepare_zip

class ProductTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root/'photos'; self.source.mkdir()
        self.photo = self.source/'two.jpg'; Image.new('RGB',(64,64)).save(self.photo)
        self.project = self.root/'project'; work=self.project/'work'; work.mkdir(parents=True)
        pd.DataFrame({'face_id':['a','b','c'],'image_path':[str(self.photo)]*3}).to_parquet(work/'faces.parquet')
        pd.DataFrame({'image_path':[str(self.photo)],'status':['ok'],'detected':[3]}).to_csv(work/'images.csv',index=False)
        pd.DataFrame({'face_id':['a','b','c'],'cluster_id':[0,1,0],'rescued':[False]*3}).to_csv(self.project/'assignments.csv',index=False)
        write_json(self.project/'project.json',{'source':str(self.source)})
        write_json(self.project/'review.json',{'revision':0,'names':{},'overrides':{}})

    def tearDown(self): self.temp.cleanup()

    def test_review_merge_split_undo_and_original_untouched(self):
        original=(self.project/'assignments.csv').read_bytes()
        edit_project(self.project,'rename',cluster=0,name='บีม')
        edit_project(self.project,'merge',source=1,target=0)
        self.assertEqual(set(catalog(self.project)[2].cluster_id),{0})
        edit_project(self.project,'undo')
        self.assertEqual(set(catalog(self.project)[2].cluster_id),{0,1})
        edit_project(self.project,'move',faces=['c'],target='new')
        self.assertEqual(set(catalog(self.project)[2].cluster_id),{0,1,2})
        self.assertEqual(catalog(self.project)[4][0],'บีม')
        self.assertEqual((self.project/'assignments.csv').read_bytes(),original)

    def test_reviewed_export_preserves_multiple_people_and_selected_only(self):
        original=self.photo.read_bytes()
        edit_project(self.project,'rename',cluster=0,name='บีม')
        out=self.root/'export'
        _,summary=export_project(self.project,out)
        self.assertEqual(summary['copies'],2)
        self.assertFalse(summary['dry_run'])
        self.assertEqual((out/'บีม/two.jpg').read_bytes(),original)
        self.assertEqual((out/'Person_ID_002/two.jpg').read_bytes(),original)
        _,summary=export_project(self.project,self.root/'selected',selected=[0])
        self.assertEqual(summary['copies'],1)
        self.assertFalse((self.root/'selected/Person_ID_002').exists())
        self.assertEqual(self.photo.read_bytes(),original)

    def test_zip_contains_person_folders_and_original_bytes_without_export_copy(self):
        edit_project(self.project,'rename',cluster=0,name='บีม')
        path,summary=prepare_zip(self.project)
        self.assertEqual(summary['copies'],2)
        with zipfile.ZipFile(path) as archive:
            self.assertEqual(archive.read('project/บีม/two.jpg'),self.photo.read_bytes())
            self.assertEqual(archive.read('project/Person_ID_002/two.jpg'),self.photo.read_bytes())
            self.assertIsNone(archive.testzip())
            self.assertEqual(len(archive.namelist()),5)
        cached,_=prepare_zip(self.project)
        self.assertEqual(path,cached)
        self.assertFalse((self.root/'export').exists())

    def test_zip_selected_person_revision_and_insufficient_space(self):
        path,_=prepare_zip(self.project,selected=[0])
        with zipfile.ZipFile(path) as archive:
            self.assertFalse(any('Person_ID_002' in name for name in archive.namelist()))
        edit_project(self.project,'rename',cluster=0,name='ใหม่')
        renamed,_=prepare_zip(self.project,selected=[0])
        self.assertNotEqual(path,renamed)
        with zipfile.ZipFile(renamed) as archive:
            self.assertIn('project/ใหม่/two.jpg',archive.namelist())
        with patch('src.product.shutil.disk_usage',return_value=SimpleNamespace(free=0)):
            with self.assertRaises(ValueError): prepare_zip(self.project,selected=[1])
        with self.assertRaises(ValueError): prepare_zip(self.project,selected=[])

    def test_automatic_same_photo_conflict_is_split_without_touching_good_groups(self):
        vectors=np.array([[1.,0],[1.,0],[1.,.01],[1.,.01],[0.,1],[0.,1]])
        labels=np.array([0,0,0,0,1,1])
        faces=pd.DataFrame({'image_path':['same','same','b','c','d','e']})
        result=split_photo_conflicts(vectors,labels,faces)
        self.assertNotEqual(result[0],result[1])
        self.assertTrue(np.array_equal(result[4:],labels[4:]))
        for group in set(result)-{-1}:
            self.assertFalse(faces[result==group].image_path.duplicated().any())

    def test_disk_preflight_name_and_source_protection(self):
        with self.assertRaises(ValueError): edit_project(self.project,'rename',cluster=0,name='../escape')
        with self.assertRaises(ValueError): edit_project(self.project,'rename',cluster=0,name='Person_ID_002')
        with self.assertRaises(ValueError): edit_project(self.project,'move',faces=['no-such-face'],target=0)
        with self.assertRaises(ValueError): export_project(self.project,self.source/'bad')
        out=self.root/'too_large'
        with patch('src.product.shutil.disk_usage',return_value=SimpleNamespace(free=0)):
            with self.assertRaises(ValueError): export_project(self.project,out)
        self.assertFalse(out.exists())
        out=self.root/'good'; export_project(self.project,out)
        with self.assertRaises(FileExistsError): export_project(self.project,out)

if __name__=='__main__': unittest.main()
