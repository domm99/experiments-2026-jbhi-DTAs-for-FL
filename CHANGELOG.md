## [1.7.0](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/compare/1.6.0...1.7.0) (2026-06-24)

### Features

* implement hierarchical trigghering of retrain based on ADWIN ([5487e9a](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/5487e9a4efcdf7a7d8196b2e6962faaff673f9c6))
* implement schedule training for adaptive policies ([529f384](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/529f384e3944ec82b8ad94457d28feb8711ebf28))

### General maintenance

* **release:** update .env versions to 1.6.0 [skip ci] ([82c7759](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/82c7759691a97bf45ffde60ce1465db0cf6da8b5))

## [1.6.0](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/compare/1.5.1...1.6.0) (2026-06-24)

### Features

* add support for adaptive policies ([2257f27](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/2257f2788f462b3de7d488b3e7b2ef276ca39147))

### General maintenance

* **release:** update .env versions to 1.5.1 [skip ci] ([d6352d4](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/d6352d4f4e2a97bde52cae8a182292ae3272ae7f))

## [1.5.1](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/compare/1.5.0...1.5.1) (2026-06-24)

### Bug Fixes

* fix bugs on devices handling when they do not have enough data ([18f529a](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/18f529a0e2cadf234242d7abe72ee2f46bdb6376))

### General maintenance

* **release:** update .env versions to 1.5.0 [skip ci] ([6135eef](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/6135eefcec7bad5d78768857f41c3b9930efc2cd))

## [1.5.0](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/compare/1.4.1...1.5.0) (2026-06-24)

### Features

* add some plotters ([cfc250e](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/cfc250e67425ab799104c585ba053c90e0e32b8f))

### Bug Fixes

* remove return false in function which should return none ([56952c3](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/56952c35906f554e23fc88e4184a8a022620f191))

### General maintenance

* increase number of global round ([c53e0a5](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/c53e0a5900d2fcb00b3042824ba3ad552babcd44))
* **release:** update .env versions to 1.4.1 [skip ci] ([193c177](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/193c177e21822321d5cdd056a36155a69b8ad63b))

## [1.4.1](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/compare/1.4.0...1.4.1) (2026-06-23)

### Bug Fixes

* fix state_dict handling ([bd6f75c](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/bd6f75cec4e60d825b558b84d95921612038daf0))

### General maintenance

* **release:** update .env versions to 1.4.0 [skip ci] ([338e63b](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/338e63b53f3fd579b3bcb77906b43e00e6e8ff37))

## [1.4.0](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/compare/1.3.0...1.4.0) (2026-06-23)

### Features

* implement model sharing from FL server to DTAs to HDTs ([d5902a6](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/d5902a67221951e08c8015eb293f8e3389c9cf17))

### Bug Fixes

* adapt train method for FL execution ([dcb2762](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/dcb27621c8035605f7a4f8e1a5034e8bb35d9c55))
* add check on at least one DT active when training ([012482a](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/012482a907f0ecd50c099ee1cfbe9a91fd5a9727))
* update dts data ([db5fa11](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/db5fa114a6a92c4ad38ab83262f4459bb21477e9))

### General maintenance

* add debug print ([cead144](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/cead1441451fd1f6ec921b97795a8d38ce1d0a5e))
* **release:** update .env versions to 1.3.0 [skip ci] ([b6db32d](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/b6db32d5335a65460ab943b1d6f140f8df822d33))

## [1.3.0](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/compare/1.2.0...1.3.0) (2026-06-23)

### Features

* add fl hyperparams ([9dbac8d](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/9dbac8d3b5ccd1ec24ed2161c2a45e95af885523))
* implement FL logic in simulator ([6b27174](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/6b2717434faa348548d70089c86a89a39a4187da))
* implement random split of patients among hospitals ([283a5b7](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/283a5b70c7135332fa15e7404d628b4292d98d83))

### Bug Fixes

* fix imports ([0c712b0](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/0c712b076eb0a9c1b3493df39f9fb711fa667e74))

### General maintenance

* **release:** update .env versions to 1.2.0 [skip ci] ([3f2d4c1](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/3f2d4c1b2796a97c758860ed9f4f75dac22b9c79))

### Style improvements

* remove trailing comma ([163f1d7](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/163f1d7fdca72b8d0e6877bc32e8fc9621e9014e))

## [1.2.0](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/compare/1.1.0...1.2.0) (2026-06-23)

### Features

* add inference monitor ([4996be0](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/4996be05db726f8c63d58539acbcdf0f2e386dca))
* implement FL server with FedAvg ([cc69aeb](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/cc69aeb8ff137004db430387c9cc68bb38dbd964))

### Bug Fixes

* add PeriodicInferenceMonitor import ([d0025ed](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/d0025ed47da164a3a24a985d12846888be5038c0))
* fix imports ([6c1f8ef](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/6c1f8ef3e404865a4ce399c58e85faca9e5c2488))
* fix imports ([4ed6357](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/4ed635772f458cb2e7f00c8e727c203ffa5fee71))
* remove codecarbon import ([6557d3b](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/6557d3b147d901aa68dc37d7b7bfbc0a96aa4b4b))

### General maintenance

* **release:** update .env versions to 1.1.0 [skip ci] ([ee3ea22](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/ee3ea22f7a9c9d28ce24d3067d7168f67ba5f35f))

## [1.1.0](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/compare/1.0.0...1.1.0) (2026-06-23)

### Features

* add DTA ([92cd1f8](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/92cd1f8f0add36403c6c702ed5a4952a5a2148d5))
* add HDT ([4c9f14a](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/4c9f14a0ef1d91122ca1959d4d8993a338bc81de))

### Bug Fixes

* fix project name in release config ([e513cf3](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/e513cf3c0992a2f027fd0644a3ad2313e8eebe53))

### General maintenance

* **release:** update .env versions to 1.0.0 [skip ci] ([e26213d](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/e26213dcfc0edfd8a464744f3d930f2afe8a5eff))

## 1.0.0 (2026-06-23)

### Features

* add learning configs ([2419f72](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/2419f723f3fbb9348bc91d213d47e86dd87f7da6))
* add learning utils ([eb925d0](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/eb925d0e7abcddb48e7d92baccf661ae0ad8e0e6))
* add simulator ([4bf22aa](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/4bf22aae1f47198a7e0e1bb56fcf407e0ab5c26a))
* setup main for RetrainAfterTime experiment ([cafedac](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/cafedacec4dc0ee8491df1ea1d27dabab1a3962b))

### Dependency updates

* **deps:** add requirements ([8b6e75b](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/8b6e75b426e2294a9c1ac65eb31689d2d6e73835))

### General maintenance

* fix project name ([415e8c5](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/415e8c50b54adb72963cd2cc896c2e5db1e107fb))
* ignoring dataset ([8e4c5df](https://github.com/domm99/experiments-2026-jbhi-DTAs-for-FL/commit/8e4c5dfeb95bfb5b48180462d4ad836508912e11))
