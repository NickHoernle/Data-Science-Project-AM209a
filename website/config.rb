###
# Compass
###

# Change Compass configuration
# compass_config do |config|
#   config.output_style = :compact
# end
set :haml, { :ugly => true, :format => :html5 }

###
# Page options, layouts, aliases and proxies
###

# Per-page layout changes:
#
# With no layout
# page "/path/to/file.html", :layout => false
#
# With alternative layout
# page "/path/to/file.html", :layout => :otherlayout
#
# A path which all have the same layout
# with_layout :admin do
#   page "/admin/*"
# end

# Proxy pages (https://middlemanapp.com/advanced/dynamic_pages/)
# proxy "/this-page-has-no-template.html", "/template-file.html", :locals => {
#  :which_fake_page => "Rendering a fake page with a local variable" }

###
# Helpers
###

# Automatic image dimensions on image_tag helper
# activate :automatic_image_sizes

# Reload the browser automatically whenever files change
# configure :development do
#   activate :livereload
# end

# Methods defined in the helpers block are available in templates
# helpers do
#   def some_helper
#     "Helping"
#   end
# end

set :css_dir, 'stylesheets'

set :js_dir, 'javascripts'

set :images_dir, 'images'

sprockets.append_path File.join root, 'bower_components'

sprockets.import_asset 'bootstrap/fonts/glyphicons-halflings-regular.woff' do |p|
  "#{fonts_dir}/glyphicons-halflings-regular.woff"
end

sprockets.import_asset 'bootstrap/fonts/glyphicons-halflings-regular.woff2' do |p|
  "#{fonts_dir}/glyphicons-halflings-regular.woff2"
end

ignore 'example-without-middleman.html'

# Build-specific configuration
configure :build do
  # For example, change the Compass output style for deployment
  # activate :minify_css

  # Minify Javascript on build
  # activate :minify_javascript

  # Enable cache buster
  # activate :asset_hash

  # Use relative URLs
  activate :relative_assets

  # Or use a different image path
  # set :http_prefix, "/Content/images/"
end

helpers do
  def references
    {
      'yelp' => "<a href='https://www.yelp.com/dataset_challenge' target='_blank'>The Yelp Dataset Challenge</a>. Accessed 2016.",
      'koren' => "Yehuda Koren, <a href='http://www.netflixprize.com/assets/GrandPrize2009_BPC_BellKor.pdf' target='_blank'>The BellKor Solution to the Netflix Grand Prize</a>. 2009.",
      'koren2' => "Yehuda Koren, <a href='http://www.cs.rochester.edu/twiki/pub/Main/HarpSeminar/Factorization_Meets_the_Neighborhood-_a_Multifaceted_Collaborative_Filtering_Model.pdf'>Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model</a>. Proc. 14th ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining (KDD'08), pp. 426–434, 2008."
    }
  end

  def render_references
    return "<ol>#{references.map{|k,v| "<li id='ref-#{k}'>#{v}</li>"}.join("\n")}</ol>".html_safe
  end

  def cite(ref)
    return "[<a href='#ref-#{ref}'>#{references.keys.index(ref)+1}</a>]".html_safe
  end
end

Dotenv.load

if ENV.key?('S3_BUCKET')
  activate :s3_sync do |s3_sync|
    s3_sync.bucket                     = ENV['S3_BUCKET']
    s3_sync.region                     = ENV['S3_REGION']
    s3_sync.aws_access_key_id          = ENV['AWS_ACCESS_KEY_ID']
    s3_sync.aws_secret_access_key      = ENV['AWS_SECRET_ACCESS_KEY']
    s3_sync.delete                     = false # We delete stray files by default.
    s3_sync.after_build                = false # We do not chain after the build step by default.
    s3_sync.prefer_gzip                = true
    s3_sync.path_style                 = true
    s3_sync.reduced_redundancy_storage = false
    s3_sync.acl                        = 'public-read'
    s3_sync.encryption                 = false
    s3_sync.version_bucket             = false
  end
end
