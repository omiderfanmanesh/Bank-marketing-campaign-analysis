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
        self.dist_by_col(col='job')
        self.dist_by_col(col='job', trans=TransformersType.LOG)

    def marital(self):
        self.box_by_col(col='age')
        self.box_by_col(col='age', trans=TransformersType.LOG)

        self.dist_by_col(col='age')
        self.dist_by_col(col='age', trans=TransformersType.LOG)

        self.violin_by_col(col='age')
        self.violin_by_col(col='age', trans=TransformersType.LOG)

    def education(self):
        self.box_by_col(col='age')
        self.box_by_col(col='age', trans=TransformersType.LOG)

        self.dist_by_col(col='age')
        self.dist_by_col(col='age', trans=TransformersType.LOG)

        self.violin_by_col(col='age')
        self.violin_by_col(col='age', trans=TransformersType.LOG)

    def default(self):
        self.box_by_col(col='age')
        self.box_by_col(col='age', trans=TransformersType.LOG)

        self.dist_by_col(col='age')
        self.dist_by_col(col='age', trans=TransformersType.LOG)

        self.violin_by_col(col='age')
        self.violin_by_col(col='age', trans=TransformersType.LOG)

    def balance(self):
        self.box_by_col(col='age')
        self.box_by_col(col='age', trans=TransformersType.LOG)

        self.dist_by_col(col='age')
        self.dist_by_col(col='age', trans=TransformersType.LOG)

        self.violin_by_col(col='age')
        self.violin_by_col(col='age', trans=TransformersType.LOG)

    def housing(self):
        self.box_by_col(col='age')
        self.box_by_col(col='age', trans=TransformersType.LOG)

        self.dist_by_col(col='age')
        self.dist_by_col(col='age', trans=TransformersType.LOG)

        self.violin_by_col(col='age')
        self.violin_by_col(col='age', trans=TransformersType.LOG)

    def loan(self):
        self.box_by_col(col='age')
        self.box_by_col(col='age', trans=TransformersType.LOG)

        self.dist_by_col(col='age')
        self.dist_by_col(col='age', trans=TransformersType.LOG)

        self.violin_by_col(col='age')
        self.violin_by_col(col='age', trans=TransformersType.LOG)

    def contact(self):
        self.box_by_col(col='age')
        self.box_by_col(col='age', trans=TransformersType.LOG)

        self.dist_by_col(col='age')
        self.dist_by_col(col='age', trans=TransformersType.LOG)

        self.violin_by_col(col='age')
        self.violin_by_col(col='age', trans=TransformersType.LOG)

    def day(self):
        self.box_by_col(col='age')
        self.box_by_col(col='age', trans=TransformersType.LOG)

        self.dist_by_col(col='age')
        self.dist_by_col(col='age', trans=TransformersType.LOG)

        self.violin_by_col(col='age')
        self.violin_by_col(col='age', trans=TransformersType.LOG)

    def month(self):
        self.box_by_col(col='age')
        self.box_by_col(col='age', trans=TransformersType.LOG)

        self.dist_by_col(col='age')
        self.dist_by_col(col='age', trans=TransformersType.LOG)

        self.violin_by_col(col='age')
        self.violin_by_col(col='age', trans=TransformersType.LOG)

    def duration(self):
        self.box_by_col(col='duration')
        self.box_by_col(col='duration', trans=TransformersType.LOG)

        self.dist_by_col(col='duration')
        self.dist_by_col(col='duration', trans=TransformersType.LOG)

        self.violin_by_col(col='duration')
        self.violin_by_col(col='duration', trans=TransformersType.LOG)

    def campaign(self):
        self.box_by_col(col='age')
        self.box_by_col(col='age', trans=TransformersType.LOG)

        self.dist_by_col(col='age')
        self.dist_by_col(col='age', trans=TransformersType.LOG)

        self.violin_by_col(col='age')
        self.violin_by_col(col='age', trans=TransformersType.LOG)

    def pdays(self):
        self.box_by_col(col='age')
        self.box_by_col(col='age', trans=TransformersType.LOG)

        self.dist_by_col(col='age')
        self.dist_by_col(col='age', trans=TransformersType.LOG)

        self.violin_by_col(col='age')
        self.violin_by_col(col='age', trans=TransformersType.LOG)

    def previous(self):
        self.box_by_col(col='age')
        self.box_by_col(col='age', trans=TransformersType.LOG)

        self.dist_by_col(col='age')
        self.dist_by_col(col='age', trans=TransformersType.LOG)

        self.violin_by_col(col='age')
        self.violin_by_col(col='age', trans=TransformersType.LOG)

    def poutcome(self):
        self.box_by_col(col='age')
        self.box_by_col(col='age', trans=TransformersType.LOG)

        self.dist_by_col(col='age')
        self.dist_by_col(col='age', trans=TransformersType.LOG)

        self.violin_by_col(col='age')
        self.violin_by_col(col='age', trans=TransformersType.LOG)
