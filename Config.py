class file_offline_train_line():
    def __init__(self):
        self.dic={}
        self.dic["User_id"] = 0
        self.dic["Merchant_id"] = 1
        self.dic["Coupon_id"] = 2
        self.dic["Discount_rate"] = 3
        self.dic["Distance"] = 4
        self.dic["Date_received"] = 5
        self.dic["Date"] = 6
class file_online_train_line():
    def __init__(self):
        self.dic={}
        self.dic["User_id"] = 0
        self.dic["Merchant_id"] = 1
        self.dic["Action"] = 2
        self.dic["Coupon_id"] = 3
        self.dic["Discount_rate"] = 4
        self.dic["Date_received"] = 5
        self.dic["Date"] = 6
class file_offline_test_line():
    def __init__(self):
        self.dic={}
        self.dic["User_id"] = 0
        self.dic["Merchant_id"] = 1
        self.dic["Coupon_id"] = 2
        self.dic["Discount_rate"] = 3
        self.dic["Distance"] = 4
        self.dic["Date_received"] = 5