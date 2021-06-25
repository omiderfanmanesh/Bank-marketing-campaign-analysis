#  Copyright (c) 2021, Omid Erfanmanesh, All rights reserved.

from data.based import TransformersType
from eda.based import BasedPlot


class BankPlots(BasedPlot):
    def __init__(self, cfg, dataset):
        super(BankPlots, self).__init__(dataset=dataset, cfg=cfg)

    def age(self):
        self.box_by_col(col='age')
        self.box_by_col(col='age', trans=TransformersType.LOG)

        self.dist_by_col(col='age')
        self.dist_by_col(col='age', trans=TransformersType.LOG)

        self.violin_by_col(col='age')
        self.violin_by_col(col='age', trans=TransformersType.LOG)

    def job(self):
        self.category_count(col='job', lbl_rotation=90)

    def marital(self):
        self.category_count(col='marital')

    def education(self):
        self.category_count(col='education')
        self.bar(x='education', y='age')



    def default(self):
        self.category_count(col='default')

    def balance(self):
        self.box_by_col(col='balance')
        self.box_by_col(col='balance', trans=TransformersType.LOG)

        self.dist_by_col(col='balance')
        self.dist_by_col(col='balance', trans=TransformersType.LOG)

    def housing(self):
        self.category_count(col='housing')

    def loan(self):
        self.category_count(col='loan')

    def contact(self):
        self.category_count(col='contact')

    def day(self):
        self.box_by_col(col='day')
        self.box_by_col(col='day', trans=TransformersType.LOG)

        self.dist_by_col(col='day')
        self.dist_by_col(col='day', trans=TransformersType.LOG)

        self.violin_by_col(col='day')
        self.violin_by_col(col='day', trans=TransformersType.LOG)

    def month(self):
        self.category_count(col='month')

    def duration(self):
        self.box_by_col(col='duration')
        self.box_by_col(col='duration', trans=TransformersType.LOG)

        self.dist_by_col(col='duration')
        self.dist_by_col(col='duration', trans=TransformersType.LOG)

        self.violin_by_col(col='duration')
        self.violin_by_col(col='duration', trans=TransformersType.LOG)

    def campaign(self):
        self.box_by_col(col='campaign')
        self.box_by_col(col='campaign', trans=TransformersType.LOG)

        self.dist_by_col(col='campaign')
        self.dist_by_col(col='campaign', trans=TransformersType.LOG)

        self.violin_by_col(col='campaign')
        self.violin_by_col(col='campaign', trans=TransformersType.LOG)

    def pdays(self):
        self.box_by_col(col='pdays')
        self.box_by_col(col='pdays', trans=TransformersType.LOG)

        self.dist_by_col(col='pdays')
        self.dist_by_col(col='pdays', trans=TransformersType.LOG)

        self.violin_by_col(col='pdays')
        self.violin_by_col(col='pdays', trans=TransformersType.LOG)

    def previous(self):
        self.box_by_col(col='previous')
        self.box_by_col(col='previous', trans=TransformersType.LOG)

        self.dist_by_col(col='previous')
        self.dist_by_col(col='previous', trans=TransformersType.LOG)

        self.violin_by_col(col='previous')
        self.violin_by_col(col='previous', trans=TransformersType.LOG)

    def poutcome(self):
        self.category_count(col='poutcome')

    def y(self):
        self.category_count(col='y')
