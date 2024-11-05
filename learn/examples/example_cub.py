import cv2 as cv2
import torch
from torch import nn
from torchvision.transforms import transforms
from networks.resnet import ResNet50

classes = ['BlackFootedAlbatross', 'LaysanAlbatross', 'SootyAlbatross', 'GrooveBilledAni', 'CrestedAuklet',
           'LeastAuklet', 'ParakeetAuklet', 'RhinocerosAuklet', 'BrewerBlackbird', 'RedWingedBlackbird',
           'RustyBlackbird', 'YellowHeadedBlackbird', 'Bobolink', 'IndigoBunting', 'LazuliBunting', 'PaintedBunting',
           'Cardinal', 'SpottedCatbird', 'GrayCatbird', 'YellowBreastedChat', 'EasternTowhee', 'ChuckWillWidow',
           'BrandtCormorant', 'RedFacedCormorant', 'PelagicCormorant', 'BronzedCowbird', 'ShinyCowbird', 'BrownCreeper',
           'AmericanCrow', 'FishCrow', 'BlackBilledCuckoo', 'MangroveCuckoo', 'YellowBilledCuckoo',
           'GrayCrownedRosyFinch', 'PurpleFinch', 'NorthernFlicker', 'AcadianFlycatcher', 'GreatCrestedFlycatcher',
           'LeastFlycatcher', 'OliveSidedFlycatcher', 'ScissorTailedFlycatcher', 'VermilionFlycatcher',
           'YellowBelliedFlycatcher', 'Frigatebird', 'NorthernFulmar', 'Gadwall', 'AmericanGoldfinch',
           'EuropeanGoldfinch', 'BoatTailedGrackle', 'EaredGrebe', 'HornedGrebe', 'PiedBilledGrebe', 'WesternGrebe',
           'BlueGrosbeak', 'EveningGrosbeak', 'PineGrosbeak', 'RoseBreastedGrosbeak', 'PigeonGuillemot',
           'CaliforniaGull', 'GlaucousWingedGull', 'HeermannGull', 'HerringGull', 'IvoryGull', 'RingBilledGull',
           'SlatyBackedGull', 'WesternGull', 'AnnaHummingbird', 'RubyThroatedHummingbird', 'RufousHummingbird',
           'GreenVioletear', 'LongTailedJaeger', 'PomarineJaeger', 'BlueJay', 'FloridaJay', 'GreenJay', 'DarkEyedJunco',
           'TropicalKingbird', 'GrayKingbird', 'BeltedKingfisher', 'GreenKingfisher', 'PiedKingfisher',
           'RingedKingfisher', 'WhiteBreastedKingfisher', 'RedLeggedKittiwake', 'HornedLark', 'PacificLoon', 'Mallard',
           'WesternMeadowlark', 'HoodedMerganser', 'RedBreastedMerganser', 'Mockingbird', 'Nighthawk',
           'ClarkNutcracker', 'WhiteBreastedNuthatch', 'BaltimoreOriole', 'HoodedOriole', 'OrchardOriole',
           'ScottOriole', 'Ovenbird', 'BrownPelican', 'WhitePelican', 'WesternWoodPewee', 'Sayornis', 'AmericanPipit',
           'WhipPoorWill', 'HornedPuffin', 'CommonRaven', 'WhiteNeckedRaven', 'AmericanRedstart', 'Geococcyx',
           'LoggerheadShrike', 'GreatGreyShrike', 'BairdSparrow', 'BlackThroatedSparrow', 'BrewerSparrow',
           'ChippingSparrow', 'ClayColoredSparrow', 'HouseSparrow', 'FieldSparrow', 'FoxSparrow', 'GrasshopperSparrow',
           'HarrisSparrow', 'HenslowSparrow', 'LeConteSparrow', 'LincolnSparrow', 'NelsonSharpTailedSparrow',
           'SavannahSparrow', 'SeasideSparrow', 'SongSparrow', 'TreeSparrow', 'VesperSparrow', 'WhiteCrownedSparrow',
           'WhiteThroatedSparrow', 'CapeGlossyStarling', 'BankSwallow', 'BarnSwallow', 'CliffSwallow', 'TreeSwallow',
           'ScarletTanager', 'SummerTanager', 'ArticTern', 'BlackTern', 'CaspianTern', 'CommonTern', 'ElegantTern',
           'ForstersTern', 'LeastTern', 'GreenTailedTowhee', 'BrownThrasher', 'SageThrasher', 'BlackCappedVireo',
           'BlueHeadedVireo', 'PhiladelphiaVireo', 'RedEyedVireo', 'WarblingVireo', 'WhiteEyedVireo',
           'YellowThroatedVireo', 'BayBreastedWarbler', 'BlackAndWhiteWarbler', 'BlackThroatedBlueWarbler',
           'BlueWingedWarbler', 'CanadaWarbler', 'CapeMayWarbler', 'CeruleanWarbler', 'ChestnutSidedWarbler',
           'GoldenWingedWarbler', 'HoodedWarbler', 'KentuckyWarbler', 'MagnoliaWarbler', 'MourningWarbler',
           'MyrtleWarbler', 'NashvilleWarbler', 'OrangeCrownedWarbler', 'PalmWarbler', 'PineWarbler', 'PrairieWarbler',
           'ProthonotaryWarbler', 'SwainsonWarbler', 'TennesseeWarbler', 'WilsonWarbler', 'WormEatingWarbler',
           'YellowWarbler', 'NorthernWaterthrush', 'LouisianaWaterthrush', 'BohemianWaxwing', 'CedarWaxwing',
           'AmericanThreeToedWoodpecker', 'PileatedWoodpecker', 'RedBelliedWoodpecker', 'RedCockadedWoodpecker',
           'RedHeadedWoodpecker', 'DownyWoodpecker', 'BewickWren', 'CactusWren', 'CarolinaWren', 'HouseWren',
           'MarshWren', 'RockWren', 'WinterWren', 'CommonYellowthroat']

net = ResNet50()
in_features = net.fc.in_features
output_dim = len(classes)
fc = nn.Linear(in_features, output_dim)
net.fc = fc

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)
net.load_state_dict(torch.load('./cub_200_2011_net.pth'))

data = cv2.imread(
    './data/CUB_200_2011/CUB_200_2011/train/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg',
    cv2.IMREAD_UNCHANGED)
# Convert BGR image to RGB image
data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

transform = transforms.Compose(
    # [
    #     transforms.ToTensor(),
    #     transforms.Resize((112, 112)),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ]
    [
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.4857, 0.4991, 0.4312],
                             std=[0.1824, 0.1813, 0.1932])
    ]
)

with torch.no_grad():
    tensor = transform(data)
    tensor = torch.reshape(tensor, (1, 3, 224, 224))
    # tensor = torch.reshape(tensor, (1, 3, 32, 32))

    result = net(tensor.cuda())
    print(result)
    _, predicted = torch.max(result, 1)
    print(predicted)
    print(classes[predicted[0]])
