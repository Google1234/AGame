class file_offline_train_line():
    def __init__(self):
        self.User_id=0
        self.Merchant_id=1
        self.Coupon_id=2
        self.Discount_rate=3
        self.Distance=4
        self.Date_received=5
        self.Date=6
class file_online_train_line():
    def __init__(self):
        self.User_id=0
        self.Merchant_id=1
        self.Action=2
        self.Coupon_id=3
        self.Discount_rate=4
        self.Date_received=5
        self.Date=6
class file_offline_test_line():
    def __init__(self):
        self.User_id=0
        self.Merchant_id=1
        self.Coupon_id=2
        self.Discount_rate=3
        self.Distance=4
        self.Date_received=5