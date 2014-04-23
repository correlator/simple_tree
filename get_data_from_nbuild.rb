# Ran this in nbuild-console in 3dna

number_of_activities = ActivityType.pluck(:id).max
csv = CSV.open('/tmp/scott_ml_tree_prospect_training', 'wb')
Signup.select(:id).limit(10_000).order("RANDOM()").each do |signup|
  activity_count = Array.new(number_of_activities, 0)
  pay_date = signup.invoice_payments.order(:created_at).first
  if pay_date
    first_activities = signup.activities.where("created_at < ?", pay_date.created_at)
  else
    first_activities = signup.activities.all
  end
  first_activities.each do |activity|
    activity_count[activity.activity_type_id] += 1
  end
  csv << [!pay_date.nil?] + activity_count
end
csv.close
