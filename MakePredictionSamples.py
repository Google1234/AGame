import pandas
import Config
import FeatureExtract

input_file_name=Config.path+"ccf_offline_stage1_test_revised.csv"
def make():
    rats = pandas.read_csv(input_file_name, header=None)
    samples = []
    user_feature = FeatureExtract.user_feature(
        Config.path + Config.feature_path + "User_from_offline_train.csv", False)
    merchant_feature = FeatureExtract.merchant_feature(
        Config.path + Config.feature_path + "Merchant_from_offline_train.csv", False)
    coupon_feature = FeatureExtract.coupon_feature(
        Config.path + Config.feature_path + "Coupon_from_offline_train.csv",False)
    file_format=Config.file_offline_test_line()

    print "total :", len(rats) / 1000, "*1000"
    for line in xrange(len(rats)):
            if rats.iloc[line][file_format.Distance] == "null":
                distance = Config.avr_distance
            else:
                distance = int(rats.iloc[line][file_format.Distance])
            samples.append(
                user_feature.get(rats.iloc[line][file_format.User_id])[1:] +
                merchant_feature.get(rats.iloc[line][file_format.Merchant_id])[1:] +
                coupon_feature.get(rats.iloc[line][file_format.Coupon_id])[1:] +
                [distance]
            )
            if line % 1000 == 0:
                print "Finish ", line / 1000, "*1000"
    # if tf.gfile.Exists(output_dir):
    #    tf.gfile.DeleteRecursively(output_dir)
    # tf.gfile.MakeDirs(output_dir)
    pandas.DataFrame(samples).to_csv(Config.sample_path + "prediction_samples.csv", index=None, header=None)
    print ("Finish")