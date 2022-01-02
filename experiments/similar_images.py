import sys

import pandas as pd
import numpy as np
import wandb
from sklearn.model_selection import train_test_split

from petfinder_pawpularity.config import create_config, to_flat_dict
from petfinder_pawpularity.run import train_folds, eval_model
from petfinder_pawpularity.util import add_path_to_data

source_avg_hash = [
    "c504568822c53675a4f425c8e5800a36",
    "5a642ecc14e9c57a05b8e010414011f2",
    "67e97de8ec7ddcda59a58b027263cdcc",
    "5ef7ba98fc97917aec56ded5d5c2b099",
    "902786862cbae94e890a090e5700298b",
    "3877f2981e502fe1812af38d4f511fd2",
    "6ae42b731c00756ddd291fa615c822a1",
    "43bd09ca68b3bcdc2b0c549fd309d1ba",
    "e09a818b7534422fb4c688f12566e38f",
    "b148cbea87c3dcc65a05b15f78910715",
    "bf8501acaeeedc2a421bac3d9af58bb7",
    "08440f8c2c040cf2941687de6dc5462f",
    "9a0238499efb15551f06ad583a6fa951",
    "43ab682adde9c14adb7c05435e5f2e0e",
    "01430d6ae02e79774b651175edd40842",
    "6dc1ae625a3bfb50571efedc0afc297c",
    "68e55574e523cf1cdc17b60ce6cc2f60",
    "9b3267c1652691240d78b7b3d072baf3",
    "5da97b511389a1b62ef7a55b0a19a532",
    "8ffde3ae7ab3726cff7ca28697687a42",
]

source_combined_hash = [
    "13d215b4c71c3dc603cd13fc3ec80181",
    "373c763f5218610e9b3f82b12ada8ae5",
    "5ef7ba98fc97917aec56ded5d5c2b099",
    "67e97de8ec7ddcda59a58b027263cdcc",
    "839087a28fa67bf97cdcaf4c8db458ef",
    "a8f044478dba8040cc410e3ec7514da1",
    "1feb99c2a4cac3f3c4f8a4510421d6f5",
    "264845a4236bc9b95123dde3fb809a88",
    "3c50a7050df30197e47865d08762f041",
    "def7b2f2685468751f711cc63611e65b",
    "37ae1a5164cd9ab4007427b08ea2c5a3",
    "3f0222f5310e4184a60a7030da8dc84b",
    "5a642ecc14e9c57a05b8e010414011f2",
    "c504568822c53675a4f425c8e5800a36",
    "2a8409a5f82061e823d06e913dee591c",
    "86a71a412f662212fe8dcd40fdaee8e6",
    "3c602cbcb19db7a0998e1411082c487d",
    "a8bb509cd1bd09b27ff5343e3f36bf9e",
    "0422cd506773b78a6f19416c98952407",
    "0b04f9560a1f429b7c48e049bcaffcca",
    "68e55574e523cf1cdc17b60ce6cc2f60",
    "9b3267c1652691240d78b7b3d072baf3",
    "1059231cf2948216fcc2ac6afb4f8db8",
    "bca6811ee0a78bdcc41b659624608125",
    "5da97b511389a1b62ef7a55b0a19a532",
    "8ffde3ae7ab3726cff7ca28697687a42",
    "78a02b3cb6ed38b2772215c0c0a7f78e",
    "c25384f6d93ca6b802925da84dfa453e",
    "08440f8c2c040cf2941687de6dc5462f",
    "bf8501acaeeedc2a421bac3d9af58bb7",
    "0c4d454d8f09c90c655bd0e2af6eb2e5",
    "fe47539e989df047507eaa60a16bc3fd",
    "5a5c229e1340c0da7798b26edf86d180",
    "dd042410dc7f02e648162d7764b50900",
    "871bb3cbdf48bd3bfd5a6779e752613e",
    "988b31dd48a1bc867dbc9e14d21b05f6",
    "dbf25ce0b2a5d3cb43af95b2bd855718",
    "e359704524fa26d6a3dcd8bfeeaedd2e",
    "43bd09ca68b3bcdc2b0c549fd309d1ba",
    "6ae42b731c00756ddd291fa615c822a1",
    "43ab682adde9c14adb7c05435e5f2e0e",
    "9a0238499efb15551f06ad583a6fa951",
    "a9513f7f0c93e179b87c01be847b3e4c",
    "b86589c3e85f784a5278e377b726a4d4",
    "38426ba3cbf5484555f2b5e9504a6b03",
    "6cb18e0936faa730077732a25c3dfb94",
    "589286d5bfdc1b26ad0bf7d4b7f74816",
    "cd909abf8f425d7e646eebe4d3bf4769",
    "9f5a457ce7e22eecd0992f4ea17b6107",
    "b967656eb7e648a524ca4ffbbc172c06",
    "b148cbea87c3dcc65a05b15f78910715",
    "e09a818b7534422fb4c688f12566e38f",
    "3877f2981e502fe1812af38d4f511fd2",
    "902786862cbae94e890a090e5700298b",
    "8f20c67f8b1230d1488138e2adbb0e64",
    "b190f25b33bd52a8aae8fd81bd069888",
    "221b2b852e65fe407ad5fd2c8e9965ef",
    "94c823294d542af6e660423f0348bf31",
    "2b737750362ef6b31068c4a4194909ed",
    "41c85c2c974cc15ca77f5ababb652f84",
    "01430d6ae02e79774b651175edd40842",
    "6dc1ae625a3bfb50571efedc0afc297c",
    "72b33c9c368d86648b756143ab19baeb",
    "763d66b9cf01069602a968e573feb334",
    "03d82e64d1b4d99f457259f03ebe604d",
    "dbc47155644aeb3edd1bd39dba9b6953",
    "851c7427071afd2eaf38af0def360987",
    "b49ad3aac4296376d7520445a27726de",
    "54563ff51aa70ea8c6a9325c15f55399",
    "b956edfd0677dd6d95de6cb29a85db9c",
    "87c6a8f85af93b84594a36f8ffd5d6b8",
    "d050e78384bd8b20e7291b3efedf6a5b",
    "04201c5191c3b980ae307b20113c8853",
    "16d8e12207ede187e65ab45d7def117b",
]

source_cnn = [
    "01430d6ae02e79774b651175edd40842",
    "03d375014d1a35e972dbc3d92413df46",
    "03d82e64d1b4d99f457259f03ebe604d",
    "04201c5191c3b980ae307b20113c8853",
    "0422cd506773b78a6f19416c98952407",
    "05010a08dc04beffa845696c357676df",
    "050ab8059a28a24723d621709f4db97c",
    "05c23d6c30f8d0b5534f7a9884a2c868",
    "06e6ac54d41071a5f55fc57ba28e8ce5",
    "07cb957b2ff279bbd17c49fe5f8b1d63",
    "08440f8c2c040cf2941687de6dc5462f",
    "093152325d04185ad02d288b95cf9183",
    "096f6af40ec40bea0f98ea10ccf6d876",
    "09f4b49020e55cc8f7674dbd79f2a058",
    "0a05c55ca864b667d31c80ce2c68d6b3",
    "0b04f9560a1f429b7c48e049bcaffcca",
    "0baa6f28fe400d3acfb6d65887c74228",
    "0bd5374dd7d4b950419b1a8a206e6595",
    "0c4d454d8f09c90c655bd0e2af6eb2e5",
    "0d4b796c33e07e65ec75dfd46af9df18",
    "0d6abba9e81300164592ab9a66c52f16",
    "0e31f2937c2057933ffacf919f523cb3",
    "0e7a0c25afae41e765901b9a922d0a75",
    "0ebc2711e7f178ccef170e327e6c6f45",
    "0f170bf3f7ce163b5498f2cca49f29f7",
    "0ff3882967eb0308d6a28b07a25439c1",
    "1059231cf2948216fcc2ac6afb4f8db8",
    "107032a5f7a36e227a1fa03bfb69af3c",
    "1121661f3bb675030647cd1f4c2b1f67",
    "11f376087b21b4a6897d17a6da4d8a16",
    "1284f5bfd5ae62d801ed827a53e27533",
    "13d215b4c71c3dc603cd13fc3ec80181",
    "143ecb5619c055a378fb8a51d964db42",
    "16d8e12207ede187e65ab45d7def117b",
    "174499e14528918c5bcc3475b0ef58a3",
    "184b5eaef61d1cc8f915a13d5de69d3c",
    "185143dd238ae718f7ffbc7164ceecd9",
    "189a2859d9e7c9361a59164284b6d530",
    "1a31d6c01c9503f894bab595dc04f62c",
    "1d1f4c131be21284d62204ab9b8a15d0",
    "1d5249af75580ec61c473cdfa1c1571e",
    "1d64007bcdb526b8bfa93c0d2cc499b0",
    "1e45fd8e07ab0d3dd528228897a66c4c",
    "1f03a452abe5f323d64d8438f187482b",
    "1f7c4697a9103a99fa08afaaba885e59",
    "1feb99c2a4cac3f3c4f8a4510421d6f5",
    "1feef8e7492c30adef9927392f151d4f",
    "2003690eccabb930ee712d7387466765",
    "20e0f2d5e24ed8448eb3c2d483fb23c6",
    "221b2b852e65fe407ad5fd2c8e9965ef",
    "2269fb644034eeda0439026cbfc55f24",
    "2289fdf7d53aee2e20886886bf9a682d",
    "236e1dc53b244ea624e6fbc53d3c813b",
    "2496f3f97901ceb5b2d8886b2ff66092",
    "2535e83141f8a986b608ad4a0857c948",
    "264845a4236bc9b95123dde3fb809a88",
    "2754ce7961bc46ac32f5c3907db51c84",
    "29f0668f797db7bec8e9dbebdb4a73e0",
    "2b737750362ef6b31068c4a4194909ed",
    "2bb0677b6e5024f8e374bf9e288f137b",
    "2c275f1dcc0842ceb0a84c74b03ba1a6",
    "2eff9709443c56c80f678f1c4a2a849e",
    "2f0a2bb5233f0e56ec0f4817cab3df6d",
    "30dafe3fde143cf3cf75814dbebdd6bb",
    "334aede77cf1b34281a103abcb569b23",
    "347797c18c481615251a621a0f28c73a",
    "34b71b3048be43242ca8a213064dfe2d",
    "365d8c50030e64ac8ebcb4d738876c2c",
    "366ec6fa887d349d3726c859171c3ec6",
    "36b9c4276138ccfafa79b6ccfbd69f42",
    "373c763f5218610e9b3f82b12ada8ae5",
    "38426ba3cbf5484555f2b5e9504a6b03",
    "3877f2981e502fe1812af38d4f511fd2",
    "38ca677f78e44977e5d8454bb8274bbd",
    "38d226f670dedeadaf0beebb501da2aa",
    "390d5d76ecdbc1f462d7750a89e99bcb",
    "3a5039a9c9a8bd5e3abc2335cc1ecb09",
    "3aabe9a8bae9195c5748f601a74da99b",
    "3c50a7050df30197e47865d08762f041",
    "3db2fa49d3cc3c154ff43bb1fb27c971",
    "4076695aab0270e2de160ee92ec196db",
    "40b632aa87f7882caf13a74607c0dee2",
    "41c85c2c974cc15ca77f5ababb652f84",
    "41f2c7ce82dd783c69fea060fc9ff51b",
    "43ab682adde9c14adb7c05435e5f2e0e",
    "43bd09ca68b3bcdc2b0c549fd309d1ba",
    "44630ddcacf20f502e7053e1a7a83eae",
    "4502869ec5d71d926d39aff17850df55",
    "47db2ae009656c7877407dfac9f3a49b",
    "4900e463696cc862724c3aa96b5d7487",
    "4b618cd51feea66ef7e30489f8effa36",
    "4caf3c2febb974032acb5516bcf972e5",
    "4dcfa90313634c784a9fd5fdaf88dba1",
    "4eb1c9699ffb77a8b1242b3b99d06bcd",
    "4f2d68ef1752f6b87ab4bdb7554374fc",
    "506df6a6470c293c95c82ba61e87fc9a",
    "50ef21ee35f099444e775fe6572647be",
    "5143a57743d830172f351751094c92fb",
    "51b8a49e78e04fd2ef3fef4d1f34e3a6",
    "5262ba8c006d525e9f2902122be99aca",
    "52990caff1f81426d57d11ce9f1720ca",
    "54563ff51aa70ea8c6a9325c15f55399",
    "548b4b231b0d52c4ed7ddcf83e0b6ea2",
    "5636d0d73274408016e9198fe4571b50",
    "56777798db053e223f1eadcd94ecba25",
    "56ffc7b3e96b839b3caeb0c6cab88277",
    "574356b5a2b44bfae3ce5cd44fc40fac",
    "574e8afe7bf1e6ed48178b94268f7097",
    "5772f335fb6686d90ca3143a69657831",
    "589286d5bfdc1b26ad0bf7d4b7f74816",
    "58d85cfa3304c8ec23c91f0a5571f184",
    "5906e2fe7fbbf37af6469807d604157d",
    "592fdd2282ab0dad710d51555521451e",
    "5a5c229e1340c0da7798b26edf86d180",
    "5a642ecc14e9c57a05b8e010414011f2",
    "5a81def55cb5ecdb5b9702bf2eef45dc",
    "5a9f7840edffede06e6597a08533e75a",
    "5aff34b4c901a0811ec832a6bf9d7391",
    "5b5359b9ef5e63104d46a7a4c429704f",
    "5c39866693bda99bf238d7da7319f108",
    "5da97b511389a1b62ef7a55b0a19a532",
    "5ed72e070597957197d7c2f8b54a15d6",
    "5ef7ba98fc97917aec56ded5d5c2b099",
    "5fc38e3206cdbfe2e2d689476d4b9f2c",
    "609d0c58dc6896647089635918fdfa7b",
    "6151aa1527348e306ab702241f8af48b",
    "61b2e3c70eb4d9e9d198cae78b3f80d8",
    "61c253ae56296fc56d2847d9ec46ad44",
    "6322aa006010abfa543a3ac224bb3d01",
    "63428ff74f79507ae1ca88dddb363f9f",
    "63bc56d35b4780c9c6174f098264bf55",
    "64b9cc9b1d15a7828806bc4835c79e95",
    "66222f351bf23ff0961e7b8914a3ef38",
    "66a5390456128efec86377dcd0adaf1f",
    "678359258627d0968f43d87759401634",
    "67e97de8ec7ddcda59a58b027263cdcc",
    "68e55574e523cf1cdc17b60ce6cc2f60",
    "699e4454d0c62cca7cc9cdff17a53386",
    "6a09a75b4eca168ca1cdc76a253b9c82",
    "6a51c027c0ae54dd3d4eec1f8063e2a8",
    "6ac397fdde24bc4bac813ed96fd97c5c",
    "6ae42b731c00756ddd291fa615c822a1",
    "6b9adfaddb17c064a6e1ae0cdfbbed50",
    "6c7cccacbd48914ce4ef5831c6150546",
    "6cb18e0936faa730077732a25c3dfb94",
    "6dc1ae625a3bfb50571efedc0afc297c",
    "6f6643eb51f6a9a5c81a1f04a6895041",
    "70a4625b37354fc4464d2ef185fa2843",
    "72b33c9c368d86648b756143ab19baeb",
    "7339432826a552c24acceb664066d173",
    "733a827b0d04f612c166e5defd27b4ae",
    "73e5eec192a19041e6d37b90c802cac3",
    "74a1c4f7b145201ccfceda0a530bee62",
    "763d66b9cf01069602a968e573feb334",
    "768592681e734c377a3f56dbc1cb56f4",
    "776d7ed84385d8538fd9d01f7b8ba3c7",
    "776d92567e4d173093527335741f49c6",
    "78a02b3cb6ed38b2772215c0c0a7f78e",
    "797df3217446c7e00215da9b7181044e",
    "79b6c7d6437d23ecf2a93a82c7ca9649",
    "79f4816b0262872e87d074af8d4cff6a",
    "7af48bb7190c236d8876fe529b449c3f",
    "7b79f94fa631062d89b4517bcd10bd36",
    "7c0823a883cd0bd4dd10dde7d232853b",
    "7c58a3861afde87271e79cbc3759251c",
    "7d2682432573974b88c076a0aad2a331",
    "7d3174fa969ad47439f514638fd9241f",
    "7d8c8997566df6c19754aaed54861d17",
    "819fa3d591dce53925610a0ccf89cd6e",
    "8235f9608f9fa066fa17c33a572e2963",
    "82c994fa8a442f8a15abb494cd1c9352",
    "830c21ab4dc831ab8f84e94a678e3fbd",
    "839087a28fa67bf97cdcaf4c8db458ef",
    "83f213c33ad2fbf2a59e57529ee6f0e0",
    "84ad669cc66a0b1481d5adc09510c07a",
    "851c7427071afd2eaf38af0def360987",
    "86547ebc49a622ef1ebb814f3fe93327",
    "8695b0b5d0621842defbc3ef702e82fd",
    "871bb3cbdf48bd3bfd5a6779e752613e",
    "872c3643ef186b382eb2bffc759513ce",
    "873353d8bf4e6b9379c2321b4c7e748a",
    "87c6a8f85af93b84594a36f8ffd5d6b8",
    "890adea291143a74919b7cdaa90f9e4e",
    "897d48a702c10e831811a780509342b6",
    "8ac008de200d2a3642a75b41be656e96",
    "8ac6b50cdc80aa28f81763dc2fa2570c",
    "8ad7272993761ab3a21634a2f737a9c2",
    "8e8626d0d0a5ee8834321a0ea22ff1c8",
    "8f12f92eaa9b021ba7fe9bfce7163c8a",
    "8f20c67f8b1230d1488138e2adbb0e64",
    "8f3f1d62e9020538d0069586a216411c",
    "8ffde3ae7ab3726cff7ca28697687a42",
    "902786862cbae94e890a090e5700298b",
    "9262ff92924c32b085360bcce65a2ca1",
    "94c823294d542af6e660423f0348bf31",
    "95ddd86aeb6a09e9a135b6254c8e41bd",
    "9603e2efab17df6c828c157bfeadc5ac",
    "963b6356418428427c49c465605446b3",
    "96b3d8e6a38200fc2bda4bbf80513150",
    "979e55877a095f65b40e2bff0a39c001",
    "988b31dd48a1bc867dbc9e14d21b05f6",
    "9a0238499efb15551f06ad583a6fa951",
    "9a0e3f4811b41853ff0f9024a4081cba",
    "9aa6a6702b67912a8d99e1afa5c42b9a",
    "9aeb994590aa4f449357e930d5fa083c",
    "9b3267c1652691240d78b7b3d072baf3",
    "9bb1c4e19070066815cafb1750885f40",
    "9c0150c08281a500e8b54f98a969dc1c",
    "9c31180667abe3873a93336ea2c78f1d",
    "9cfeb5d1dd7191a7925e6da0ef6f794a",
    "9d467e607c8bda258c65e4bac92b49ff",
    "9df609161675427a8f04f772f3fd7f1a",
    "9e41a3713920319fada32c073729e30c",
    "9e7ec1d9eafd5a762436e10bbda7563d",
    "9e9b0473dea85c5ccfad3df3ecca531d",
    "9f40111d360fe1360674b6f751039dab",
    "9f541f7cc2d8a0ba64d7573701564248",
    "9f54629b08637fe0eb59b349fcf526cc",
    "9f5a457ce7e22eecd0992f4ea17b6107",
    "9f61fa01a89c7147e524e3fb1c2b1e67",
    "9f9fea2b20bf746a9cd4518967bf0671",
    "a0103fe762389e790dc0dfd499f79986",
    "a130e9225d15896933de5ced1a85f78f",
    "a29a4870c0f3f0262b14e15d838ea912",
    "a3474f306d5686cd0d15442c50382a71",
    "a3ba6a28f947ba9ab32f463a48898503",
    "a7432809d92f8e6eb62da6cf824b496c",
    "a883c48295d89d73e5a2b2d41cfecc56",
    "a8a31760a4075b12318d68373cab42e1",
    "a8f044478dba8040cc410e3ec7514da1",
    "a915a67cc115f157801ff9f6b4532dd8",
    "a9513f7f0c93e179b87c01be847b3e4c",
    "abc3bc25190cf419862c6c7e10f14e77",
    "ac624a8956b4730a28b1479ea30f1c54",
    "ac8823ae3ef0da44d00c9f7fab5ed3ea",
    "acc50150d724a6cb2cd9814ea9e4b987",
    "aeb9a74db3108e287291d0bb3ac58766",
    "aedbf9fcafcf8928e49109f227014851",
    "b13ecfa2c822038095cfa8046d0f7e83",
    "b148cbea87c3dcc65a05b15f78910715",
    "b190f25b33bd52a8aae8fd81bd069888",
    "b1d2dc1fa65fe7c29da765c0ad7b2895",
    "b27d03d4a6096b09d48b480ff41da753",
    "b2a34a7a7c2ecf1e961bca0947c04997",
    "b2eed3052c0e2d8965dfea7b50d04676",
    "b49ad3aac4296376d7520445a27726de",
    "b79d4e6feff2d2cdaf86d9fe9aa9114a",
    "b86589c3e85f784a5278e377b726a4d4",
    "b8f438a9b4abec808d60b285c4e53a8b",
    "b91f69ab9dcd3302c0bc370ca0518ee1",
    "b956edfd0677dd6d95de6cb29a85db9c",
    "b967656eb7e648a524ca4ffbbc172c06",
    "ba0fcc8a37286942e2c8b37124219156",
    "bbba9168154d8e4e57548d765d6a2aa1",
    "bbf33ad286f3602dd1468007627984ad",
    "bc50340a10257dfc1f682645609b8445",
    "bca6811ee0a78bdcc41b659624608125",
    "bdc507fd18dd4d15d105f4e03d1a3639",
    "be4f1e17767afe7cbfaf1fb6334e6568",
    "be643944d1c53858a2bfbae6ae435fca",
    "bf8501acaeeedc2a421bac3d9af58bb7",
    "c00198f31a74927d061f17637c1599a6",
    "c20c5d9a1d120daaf08aa5233b8be4ac",
    "c25384f6d93ca6b802925da84dfa453e",
    "c307f87d59c523de57e913e886cde5d8",
    "c504568822c53675a4f425c8e5800a36",
    "c54843ce85be6139aa248dfc462506fd",
    "c5e453e9db92cb9ab31d1b73c31fbea1",
    "c5f729d18607cc381edfe96e0694d3b8",
    "c88c94b87ae276967221c0e7ebd79f61",
    "c935d33d30fab139b87567e6445fe2fe",
    "cca6f558ef342ba6f1c3a8ec36aa77f4",
    "cd909abf8f425d7e646eebe4d3bf4769",
    "cdce90fba529efbd06a374cbfe541fdf",
    "cf866794828008ad18c249226dc768a9",
    "d00dbffcf82860d010097201974a494c",
    "d050e78384bd8b20e7291b3efedf6a5b",
    "d08a2a68b33dbe841b96408fe5128a6f",
    "d129848143caf8b89d39afffc628f94a",
    "d14973fa69a8ddde9adf791f6ae4e25f",
    "d2207138575b7f6f6c8b8fa8a6695c5f",
    "d28a861026ac8424091b8062a00d1d0b",
    "d301063a38729fc46bd87771ae2417f6",
    "d4b3b181788ed2231dcee96da13547b3",
    "d759c249279a6cc737140741a2e71531",
    "d772302806bbcded2ffd576491cd70f4",
    "d77cbb32318b41770954ed1a73e35dcb",
    "d7d222be164177b8a1033fb51a5d762f",
    "d8571bbd0e1247f2a96e42ec82b57f9f",
    "d9d332dda88f2d3f80d336fae3d0b05d",
    "da2de5a707339d98c4477ab7a2669140",
    "dbc47155644aeb3edd1bd39dba9b6953",
    "dbf25ce0b2a5d3cb43af95b2bd855718",
    "dc967d31260453e8f7c1d2e55f6f65e5",
    "dcae448678ce2779659104d1f5970e92",
    "dd042410dc7f02e648162d7764b50900",
    "ddedcb57d7a97c245b0aeb1915923032",
    "def7b2f2685468751f711cc63611e65b",
    "e09a818b7534422fb4c688f12566e38f",
    "e15df5d9baba1776a540e3b7a9d94592",
    "e2309ff739ab2909fa854c3256406fdb",
    "e33344d67e824fa1d401b855a7255f33",
    "e359704524fa26d6a3dcd8bfeeaedd2e",
    "e5b886aacd9528df428033bf79066877",
    "eaa23ff7ed57cd3d081f77570777a495",
    "ee97d86d26506239666d3679ff4a73aa",
    "f15868f7fc13687acde228a81334df8d",
    "f1e31942da5e13026dc2c35291d6a97b",
    "f29ea6ff5dae5c0620ad8a7b687e99fd",
    "f309d97e973683bca75adfbca9dd2b63",
    "f34f5b7e71119a071c9a3c49319c57e7",
    "f3c2efc34229ba5091c4fba61957b1b8",
    "f3e499ebd75d43a3a248596153f57364",
    "f521da959d45b29cfa635cecb3a6b061",
    "f580fa86073cf7d9db37dacf80525db8",
    "f81c984b3a8151a900f9bd210649a60c",
    "f8b98cc7c821af008757b2f78fe918ad",
    "fa2041a37321946494eea40e952326bb",
    "fa45fffafa2aa7e48dbc183a99f71aaf",
    "faa96f1229079881df2f1817eb184957",
    "fbdda0afbd9a513fbb520c7e0655120e",
    "fc21df34d426af42d3eca2e52b40957e",
    "fd0f4f709cb19d395c05c46d99facc19",
    "fdb5afed6c5e05ed7a244bc70739ef86",
    "fe04b9ace6e737c31799815c927f3507",
    "fe47539e989df047507eaa60a16bc3fd",
    "fe4854282aebafd366a9fbc36364ef60",
    "fea5968d4979d8d98d964c31e0c7fe66",
    "feebe928d75782d0db1eede305fe5b41",
    "ffdf2e8673a1da6fb80342fa3b119a20",
]

source_mobilenet = [
    "0c4d454d8f09c90c655bd0e2af6eb2e5",
    "1feb99c2a4cac3f3c4f8a4510421d6f5",
    "264845a4236bc9b95123dde3fb809a88",
    "373c763f5218610e9b3f82b12ada8ae5",
    "41c85c2c974cc15ca77f5ababb652f84",
    "43bd09ca68b3bcdc2b0c549fd309d1ba",
    "54563ff51aa70ea8c6a9325c15f55399",
    "5da97b511389a1b62ef7a55b0a19a532",
    "5ef7ba98fc97917aec56ded5d5c2b099",
    "67e97de8ec7ddcda59a58b027263cdcc",
    "6ae42b731c00756ddd291fa615c822a1",
    "6cb18e0936faa730077732a25c3dfb94",
    "6dc1ae625a3bfb50571efedc0afc297c",
    "72b33c9c368d86648b756143ab19baeb",
    "763d66b9cf01069602a968e573feb334",
    "8ffde3ae7ab3726cff7ca28697687a42",
    "902786862cbae94e890a090e5700298b",
    "988b31dd48a1bc867dbc9e14d21b05f6",
    "9a0238499efb15551f06ad583a6fa951",
    "9b3267c1652691240d78b7b3d072baf3",
    "b49ad3aac4296376d7520445a27726de",
    "b86589c3e85f784a5278e377b726a4d4",
    "b956edfd0677dd6d95de6cb29a85db9c",
    "bca6811ee0a78bdcc41b659624608125",
    "c25384f6d93ca6b802925da84dfa453e",
    "c504568822c53675a4f425c8e5800a36",
    "d050e78384bd8b20e7291b3efedf6a5b",
    "dbc47155644aeb3edd1bd39dba9b6953",
    "dbf25ce0b2a5d3cb43af95b2bd855718",
    "dd042410dc7f02e648162d7764b50900",
    "e09a818b7534422fb4c688f12566e38f",
    "e359704524fa26d6a3dcd8bfeeaedd2e",
    "fe47539e989df047507eaa60a16bc3fd",
]


sources = dict(
    all=[],
    avg_hash=source_avg_hash,
    combined_hash=source_combined_hash,
    cnn=source_cnn,
    mobilenet=source_mobilenet,
)


def remove_duplicates(data, remove_ids):
    remove_ids = np.array(remove_ids)
    original_ids = data["Id"].to_numpy()
    is_remove = original_ids.reshape(-1, 1) == remove_ids.reshape(1, -1)
    removed_idx = np.sum(is_remove, axis=1) == 0
    return data[removed_idx]


def run_source(wandb_params, data, val_data, source_name):
    config = create_config(
        dict(
            model_name="swin_tiny_patch4_window7_224",
            source_name=source_name,
            n_folds=5,
        )
    )
    # run = wandb.init(config=to_flat_dict(config), reinit=True, **wandb_params)
    models = train_folds(config, data, wandb_params=wandb_params)

    for i, model in enumerate(models):
        print(f"=========== eval for {i} model ===========")
        config.model_idx = i
        run = wandb.init(
            config=config,
            reinit=True,
            project="uncategorized",
            entity="wangyashuu",
        )
        eval_model(config, model, val_data)
        run.finish()


def run(wandb_entity, wandb_project, wandb_mode=None):
    wandb_params = dict(
        entity=wandb_entity,
        project=wandb_project,
        mode=wandb_mode,
    )

    data_path = "../input/petfinder-pawpularity-score/train"
    data = add_path_to_data(pd.read_csv(f"{data_path}.csv"), data_path)

    train_idx, val_idx = train_test_split(
        np.arange(0, len(data)),
        test_size=0.15,
        random_state=2,
    )

    train_data, val_data = data.iloc[train_idx], data.iloc[val_idx]
    for name, source in sources.items():
        print(f"=========== train for source {name} ===========")
        cleaned_data = remove_duplicates(train_data, source)
        run_source(wandb_params, cleaned_data, val_data, name)


if __name__ == "__main__":
    wandb_entity = sys.argv[1] if len(sys.argv) > 1 else None
    wandb_project = sys.argv[2] if len(sys.argv) > 2 else None
    run(wandb_entity, wandb_project)