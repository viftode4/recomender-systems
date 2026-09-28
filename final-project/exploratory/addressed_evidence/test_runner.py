import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from exploratory.addressed_evidence import run_experiment as run
from study import digest, write_json


class RunnerTests(unittest.TestCase):
    def setUp(self):
        run.deterministic_runtime()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root/'source'
        self.source.mkdir()
        self.users, self.items = [f'u{i}' for i in range(8)], ['[PAD]']+[f'i{i}' for i in range(24)]
        self.pairs = {'train': [], 'valid': [], 'test': []}
        rows = []
        for u, user in enumerate(self.users):
            for j in range(7):
                item = self.items[1+(2*u+j)%24]
                part = 'train' if j < 5 else 'valid' if j == 5 else 'test'
                self.pairs[part].append((user, item))
                rows.append((user, item, str(1+(u+j)%5) if part != 'test' else 'UNPARSED_TEST_CATEGORY'))
        for part, pairs in self.pairs.items():
            (self.source/f'{part}.tsv').write_text('user_id\titem_id\n'+''.join(f'{u}\t{i}\n' for u,i in pairs))
        self.ratings = self.root/'tiny.inter'
        self.ratings.write_text('user_id:token\titem_id:token\trating:float\n'+''.join(f'{u}\t{i}\t{r}\n' for u,i,r in rows))
        self.metadata = self.root/'tiny.item'
        self.metadata.write_text('item_id:token\tclass:token_seq\n'+''.join(f'{i}\tDrama\n' for i in self.items[1:]))
        write_json(self.source/'ids.json', {'padding_index':0, 'users':['[PAD]']+self.users, 'items':self.items})
        np.savez(self.source/'valid-scores.npz', users=self.users, items=self.items,
                 scores=np.zeros((len(self.users),len(self.items))))
        write_json(self.source/'manifest.json', {'status':'complete', 'test_evaluated':False,
            'validation_used_for_training':False, 'data_sha256':{p.name:digest(p) for p in (self.ratings,self.metadata)},
            'split_sha256':{part:digest(self.source/f'{part}.tsv') for part in self.pairs}})
        self.sources = patch.object(run, 'source_hashes', return_value={'synthetic.py':'fixed'})
        self.config = patch.object(run, 'MODEL_CONFIG', {'k':3,'init_scale':.01,'pair_bound':1.})
        self.sources.start(); self.config.start()

    def tearDown(self):
        self.config.stop(); self.sources.stop()
        self.temp.cleanup()

    def test_benchmark_joins_train_only_and_never_opens_validation_or_test_split(self):
        # Even malformed validation categories must not be parsed by benchmark.
        text = self.ratings.read_text().splitlines()
        validation = set(self.pairs['valid'])
        self.ratings.write_text('\n'.join(line.rsplit('\t',1)[0]+'\tUNPARSED_VALID_CATEGORY'
            if tuple(line.split('\t')[:2]) in validation else line for line in text)+'\n')
        manifest = json.loads((self.source/'manifest.json').read_text())
        manifest['data_sha256'][self.ratings.name] = digest(self.ratings)
        write_json(self.source/'manifest.json',manifest)
        (self.source/'valid-scores.npz').unlink()
        original_open = Path.open
        def guarded(path,*args,**kwargs):
            if path.name in ('valid.tsv','test.tsv','valid-scores.npz'):
                raise AssertionError('Benchmark opened forbidden split')
            return original_open(path,*args,**kwargs)
        with patch.object(Path,'open',guarded):
            result = run.benchmark(self.source,self.ratings,self.root/'benchmark.json',7)
        self.assertFalse(result['validation_read'])
        self.assertFalse(result['test_read'])
        self.assertEqual(result['variants']['additive']['initial_state_sha256'], result['variants']['pair']['initial_state_sha256'])
        self.assertEqual(result['variants']['pair']['training_diagnostic']['epoch'],1)

    def test_saved_model_replays_exactly_and_meta_selection_ignores_poisoned_development(self):
        loaded = run.load_source(self.source,self.ratings,self.metadata)
        users,items,_,_,training,validation = loaded[:6]
        matrix = run.categorical_matrix(users,items,training)
        neighbors = run.build_neighbors(matrix,k=3)
        fit,dev = run.partition_users(users,7)
        poisoned = {pair:(value if pair[0] in fit else np.nan) for pair,value in validation.items()}
        budget = run.training_budget(2,[1,2])
        results = []
        with redirect_stdout(io.StringIO()):
            for name,labels in [('one',validation),('two',poisoned)]:
                folder=self.root/name; folder.mkdir()
                results.append(run.train_variant(matrix,neighbors,users,items,labels,fit,7,'pair',folder,budget))
        self.assertEqual(results[0]['epoch'],results[1]['epoch'])
        self.assertEqual(results[0]['logits_array_sha256'],results[1]['logits_array_sha256'])
        self.assertTrue(results[0]['checkpoint_replay_exact'])
        model,checkpoint=run.restore(self.root/'one'/results[0]['checkpoint'])
        self.assertEqual(run.array_digest(run.predict_logits(model,matrix)),results[0]['logits_array_sha256'])
        self.assertEqual(checkpoint['config']['variant'],'pair')

    def test_run_persists_both_choices_before_development_and_never_opens_test(self):
        output=self.root/'research'/'7'
        original_evaluate=run.evaluate_logits
        calls=[]
        def checked(*args,**kwargs):
            selections=json.loads((output/'selection.json').read_text())
            self.assertEqual(set(selections),set(run.VARIANTS))
            self.assertTrue(all(row['checkpoint_replay_exact'] for row in selections.values()))
            calls.append(1)
            return original_evaluate(*args,**kwargs)
        original_open=Path.open
        def guarded(path,*args,**kwargs):
            if path.name=='test.tsv':
                raise AssertionError('Research opened TEST')
            return original_open(path,*args,**kwargs)
        with patch.object(Path,'open',guarded), patch.object(run,'evaluate_logits',side_effect=checked), redirect_stdout(io.StringIO()):
            result=run.run_seed(self.source,self.ratings,self.metadata,output,7,run.training_budget(2,[1,2]))
        self.assertEqual(len(calls),2)
        self.assertEqual(result['stage'],'reused_development')
        self.assertFalse(result['fresh_confirmation'])
        self.assertFalse(result['test_read'])
        manifest=json.loads((output/'manifest.json').read_text())
        self.assertTrue(manifest['dataset_test_previously_evaluated'])
        for name,expected in manifest['output_sha256'].items():
            self.assertEqual(digest(output/name),expected)
        evidence=self.root/'evidence'
        run.curate(output.parent,evidence,[7])
        markdown=(evidence/'RESULTS.md').read_text()
        self.assertIn('| 7 | additive |',markdown)
        self.assertIn('| 7 | pair |',markdown)
        self.assertIn('Descriptive equal-seed mean paired NLL difference',markdown)
        for name,expected in json.loads((evidence/'SHA256.json').read_text()).items():
            self.assertEqual(digest(evidence/name),expected)
        self.assertNotIn('per_user',(evidence/'aggregates.json').read_text())

    def test_budget_extension_is_explicit_and_selected_minimum_differs_from_patience_threshold(self):
        budget=run.training_budget(2000,patience=100)
        self.assertEqual(budget['checkpoints'][:5],[10,30,60,100,120])
        self.assertEqual(budget['checkpoints'][-1],2000)
        users,items,matrix,_=run.load_training_only(self.source,self.ratings)
        neighbors=run.build_neighbors(matrix,k=3)
        folder=self.root/'stop';folder.mkdir()
        budget=run.training_budget(4,[1,2,3,4],patience=1,min_improvement=.001)
        with patch.object(run,'joint_metrics',side_effect=[{'macro_joint_nll':5.},{'macro_joint_nll':4.9999}]), redirect_stdout(io.StringIO()):
            selected=run.train_variant(matrix,neighbors,users,items,{},set(users[:4]),7,'additive',folder,budget)
        self.assertEqual(selected['trained_epochs'],2)
        self.assertEqual(selected['epoch'],2)
        self.assertEqual(selected['stop_reason'],'predeclared_patience')

    def test_benchmark_source_change_rejected_without_export(self):
        with patch.object(run,'source_hashes',side_effect=[{'source':'before'},{'source':'after'}]):
            with self.assertRaisesRegex(ValueError,'sources changed'):
                run.benchmark(self.source,self.ratings,self.root/'rejected.json',7)
        self.assertFalse((self.root/'rejected.json').exists())

    def test_interrupted_optimizer_resume_matches_uninterrupted_fit_exactly(self):
        loaded=run.load_source(self.source,self.ratings,self.metadata)
        users,items,_,_,training,labels=loaded[:6]
        matrix=run.categorical_matrix(users,items,training)
        neighbors=run.build_neighbors(matrix,k=3)
        fit,_=run.partition_users(users,7)
        budget=run.training_budget(4,[1,2,3,4],patience=0)
        complete,interrupted=self.root/'complete',self.root/'interrupted'
        complete.mkdir();interrupted.mkdir()
        original_epoch=run.train_epoch
        def stop_before_third(model,optimizer,matrix,seed,epoch):
            if epoch==3:
                raise InterruptedError('synthetic interruption')
            return original_epoch(model,optimizer,matrix,seed,epoch)
        with redirect_stdout(io.StringIO()):
            first=run.train_variant(matrix,neighbors,users,items,labels,fit,7,'pair',complete,budget)
            with patch.object(run,'train_epoch',side_effect=stop_before_third):
                with self.assertRaises(InterruptedError):
                    run.train_variant(matrix,neighbors,users,items,labels,fit,7,'pair',interrupted,budget)
            second=run.train_variant(matrix,neighbors,users,items,labels,fit,7,'pair',interrupted,budget,resume=True)
        self.assertEqual(first['logits_array_sha256'],second['logits_array_sha256'])
        self.assertEqual(first['epoch'],second['epoch'])
        for filename in ('training-trace.json','selection-grid.json'):
            self.assertEqual(json.loads((complete/filename).read_text()),json.loads((interrupted/filename).read_text()))
        self.assertEqual({p.name for p in interrupted.glob('*.pt')},{'best.pt','latest.pt'})
        self.assertTrue(second['resumed'])

    def test_resume_rejects_changed_budget_and_optimizer_corruption(self):
        loaded=run.load_source(self.source,self.ratings,self.metadata)
        users,items,_,_,training,labels=loaded[:6]
        matrix=run.categorical_matrix(users,items,training)
        neighbors=run.build_neighbors(matrix,k=3)
        fit,_=run.partition_users(users,7)
        folder=self.root/'resume-guards';folder.mkdir()
        budget=run.training_budget(2,[1,2],patience=0)
        with redirect_stdout(io.StringIO()):
            run.train_variant(matrix,neighbors,users,items,labels,fit,7,'pair',folder,budget)
        with self.assertRaisesRegex(ValueError,'seal differs'):
            run.train_variant(matrix,neighbors,users,items,labels,fit,7,'pair',folder,
                              run.training_budget(3,[1,2,3],patience=0),resume=True)
        checkpoint=torch.load(folder/'latest.pt',weights_only=True)
        next(iter(checkpoint['optimizer']['state'].values()))['exp_avg'].add_(1)
        torch.save(checkpoint,folder/'latest.pt')
        with self.assertRaisesRegex(ValueError,'seal differs'):
            run.train_variant(matrix,neighbors,users,items,labels,fit,7,'pair',folder,budget,resume=True)

    def test_resumed_replay_checks_independently_stored_best_logit_digest(self):
        loaded=run.load_source(self.source,self.ratings,self.metadata)
        users,items,_,_,training,labels=loaded[:6]
        matrix=run.categorical_matrix(users,items,training)
        neighbors=run.build_neighbors(matrix,k=3)
        fit,_=run.partition_users(users,7)
        folder=self.root/'bad-replay';folder.mkdir()
        budget=run.training_budget(1,[1],patience=0)
        with redirect_stdout(io.StringIO()):
            run.train_variant(matrix,neighbors,users,items,labels,fit,7,'pair',folder,budget)
        checkpoint=torch.load(folder/'latest.pt',weights_only=True)
        checkpoint.pop('integrity_sha256')
        checkpoint['best_logits_array_sha256']='wrong'
        checkpoint['integrity_sha256']=run.tree_digest(checkpoint)
        torch.save(checkpoint,folder/'latest.pt')
        with self.assertRaisesRegex(AssertionError,'best-logit digest'):
            run.train_variant(matrix,neighbors,users,items,labels,fit,7,'pair',folder,budget,resume=True)

    def test_run_resume_rejects_changed_raw_source_before_overwriting_saved_inputs(self):
        output=self.root/'interrupted-run'
        budget=run.training_budget(1,[1],patience=0)
        with patch.object(run,'train_variant',side_effect=InterruptedError('synthetic interruption')):
            with self.assertRaises(InterruptedError):
                run.run_seed(self.source,self.ratings,self.metadata,output,7,budget)
        preserved=digest(output/'training-categories.npz')
        source_manifest=json.loads((self.source/'manifest.json').read_text())
        source_manifest['extra_metadata']='changed after interruption'
        write_json(self.source/'manifest.json',source_manifest)
        with self.assertRaisesRegex(ValueError,'raw-input'):
            run.run_seed(self.source,self.ratings,self.metadata,output,7,budget,resume=True)
        self.assertEqual(digest(output/'training-categories.npz'),preserved)


if __name__=='__main__':
    unittest.main()
